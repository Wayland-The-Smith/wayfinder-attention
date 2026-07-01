"""Multi-layer query-conditioned pointer heads for slot-pointer training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPointerMLPHead(nn.Module):
    """
    Sum query–key dot products across transformer layers → per-position scores (B, T),
    pad to ``max_seq_len``, expand with a small MLP, then compress to slot or full logits.
    """

    def __init__(
        self,
        *,
        max_seq_len: int,
        mlp_hidden: int,
        num_slot_outputs: int,
        d_model: int,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_slot_outputs = num_slot_outputs
        self.d_model = d_model
        self.expand = nn.Linear(max_seq_len, mlp_hidden)
        self.compress_slots = nn.Linear(mlp_hidden, num_slot_outputs)
        self.compress_full = nn.Linear(mlp_hidden, max_seq_len)

    def layer_position_scores(
        self,
        layer_hidden_states: list[torch.Tensor],
        question_index: torch.Tensor,
    ) -> torch.Tensor:
        if not layer_hidden_states:
            raise ValueError("layer_hidden_states must be non-empty")
        B, T, _ = layer_hidden_states[0].shape
        batch_idx = torch.arange(B, device=layer_hidden_states[0].device)
        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.zeros(B, T, device=layer_hidden_states[0].device, dtype=layer_hidden_states[0].dtype)
        for hidden in layer_hidden_states:
            h_q = hidden[batch_idx, question_index]
            scores = scores + torch.einsum("bd,btd->bt", h_q, hidden) * scale
        return scores

    def forward(
        self,
        layer_hidden_states: list[torch.Tensor],
        question_index: torch.Tensor,
        *,
        mode: str,
    ) -> torch.Tensor:
        scores = self.layer_position_scores(layer_hidden_states, question_index)
        B, T = scores.shape
        padded = scores.new_zeros(B, self.max_seq_len)
        copy_len = min(T, self.max_seq_len)
        padded[:, :copy_len] = scores[:, :copy_len]
        hidden = F.gelu(self.expand(padded))
        if mode == "value_slots":
            return self.compress_slots(hidden)
        full_logits = self.compress_full(hidden)
        return full_logits[:, :copy_len]


class QueryPoolMLPTokenHead(nn.Module):
    """
    Query-softmax gated position mix per layer → GELU → vocab logits.

    For each layer hidden (B, T, d):
      scores[t] = dot(h[question], h[t])
      h' = softmax(scores) * h
      mix: Linear(T → K) on sequence axis → (B, K, d)
    Concatenate all layers → (B, n_layers * K * d) → Linear → vocab_size.
    """

    def __init__(
        self,
        *,
        max_seq_len: int,
        d_model: int,
        vocab_size: int,
        pool_positions: int = 16,
        n_layers: int = 2,
    ):
        super().__init__()
        self.max_seq_len = int(max_seq_len)
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.pool_positions = int(pool_positions)
        self.n_layers = int(n_layers)
        self.mix_layers = nn.ModuleList(
            [nn.Linear(self.max_seq_len, self.pool_positions) for _ in range(self.n_layers)]
        )
        flat_dim = self.n_layers * self.pool_positions * self.d_model
        self.out = nn.Linear(flat_dim, self.vocab_size)

    @staticmethod
    def _query_scores(
        hidden: torch.Tensor,
        question_index: torch.Tensor,
        *,
        d_model: int,
    ) -> torch.Tensor:
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        h_q = hidden[batch_idx, question_index]
        scale = 1.0 / math.sqrt(d_model)
        return torch.einsum("bd,btd->bt", h_q, hidden) * scale

    def _mix_layer(
        self,
        hidden: torch.Tensor,
        question_index: torch.Tensor,
        mix: nn.Linear,
    ) -> torch.Tensor:
        scores = self._query_scores(hidden, question_index, d_model=self.d_model)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        gated = hidden * weights
        b, t, d = gated.shape
        padded = gated.new_zeros(b, self.max_seq_len, d)
        copy_len = min(t, self.max_seq_len)
        padded[:, :copy_len] = gated[:, :copy_len]
        mixed = mix(padded.transpose(1, 2)).transpose(1, 2)
        return F.gelu(mixed)

    def forward(
        self,
        layer_hidden_states: list[torch.Tensor],
        question_index: torch.Tensor,
    ) -> torch.Tensor:
        if len(layer_hidden_states) != self.n_layers:
            raise ValueError(
                f"expected {self.n_layers} layer hiddens, got {len(layer_hidden_states)}"
            )
        parts: list[torch.Tensor] = []
        for hidden, mix in zip(layer_hidden_states, self.mix_layers):
            parts.append(self._mix_layer(hidden, question_index, mix))
        flat = torch.cat(parts, dim=1).reshape(layer_hidden_states[0].size(0), -1)
        return self.out(flat)
