"""Small transformer language model for routing attention experiments."""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from routing_attention.models.pointer_head import MultiLayerPointerMLPHead, QueryPoolMLPTokenHead
from routing_attention.models.attention import (
    DenseAttention,
    DenseSDPAAttention,
    KeyVectorSparseAttention,
    LearnedAddressSparseAttention,
    LinearAttention,
    LocalAttention,
    RoutingSparseAttention,
)
from routing_attention.models.learned_address import LearnedAddressModule, build_address_book_from_config
from routing_attention.models.router import MultiScaleRouter, PerLayerRouter, RouterMLP, build_router_from_config


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attention_module: nn.Module,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.attn = attention_module
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_per_head: bool = False,
        **attn_kwargs,
    ):
        residual = x
        x_norm = self.ln1(x)
        if return_attention:
            attn_out, attn_w = self.attn(
                x_norm,
                attn_mask=attn_mask,
                return_attention=True,
                return_per_head=return_per_head,
                **attn_kwargs,
            )
            x = residual + attn_out
            x = x + self.ff(self.ln2(x))
            return x, attn_w
        attn_out = self.attn(x_norm, attn_mask=attn_mask, **attn_kwargs)
        x = residual + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only transformer for language modeling."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        attention_type: Literal[
            "dense", "dense_flash", "linear", "local", "routing", "key_vector", "learned_address"
        ] = "dense",
        router: Optional[nn.Module] = None,
        routing_top_k: int = 32,
        local_window: int = 64,
        pad_token_id: int = 0,
        num_digit_classes: int = 0,
        output_head: Literal["lm_token", "pointer_index", "pointer_mlp", "pool_mlp_token"] = "lm_token",
        pointer_target_mode: Literal["value_slots", "full_sequence"] = "full_sequence",
        pointer_mlp_hidden: int = 2100,
        num_pointer_slots: int = 50,
        pool_mlp_positions: int = 16,
        config: Optional[dict] = None,
        retrieval_config: Optional[dict] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_head = output_head
        self.pointer_target_mode = pointer_target_mode
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_token_id = pad_token_id
        self.attention_type = attention_type
        self.routing_top_k = routing_top_k
        self.retrieval_config = retrieval_config
        if retrieval_config is None and config is not None:
            self.retrieval_config = config.get("retrieval")

        if router is None and attention_type == "routing" and config is not None:
            router = build_router_from_config(config)

        self.router = router
        self.address_book = None
        la_cfg = (config or {}).get("learned_address", {})

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for layer_idx in range(n_layers):
            if attention_type in ("dense", "dense_flash"):
                # Fair dense baseline: always use fused SDPA (Flash when available).
                attn = DenseSDPAAttention(d_model, n_heads, dropout)
            elif attention_type == "linear":
                attn = LinearAttention(d_model, n_heads, dropout)
            elif attention_type == "local":
                attn = LocalAttention(d_model, n_heads, local_window, dropout)
            elif attention_type == "key_vector":
                attn = KeyVectorSparseAttention(
                    d_model,
                    n_heads,
                    routing_top_k,
                    dropout,
                    retrieval_config=self.retrieval_config,
                )
            elif attention_type == "learned_address":
                if self.address_book is None and config is not None:
                    self.address_book = build_address_book_from_config(config)
                addr_mod = (
                    self.address_book.get_address(layer_idx) if self.address_book is not None else None
                )
                attn = LearnedAddressSparseAttention(
                    d_model,
                    n_heads,
                    routing_top_k,
                    address_dim=la_cfg.get("address_dim", 32),
                    similarity=la_cfg.get("similarity", "asymmetric"),
                    dropout=dropout,
                    address_module=addr_mod,
                    retrieval_config=self.retrieval_config,
                )
            elif attention_type == "routing":
                if router is None:
                    router = RouterMLP(d_model, routing_dim=32)
                attn = RoutingSparseAttention(
                    d_model, n_heads, router, routing_top_k, dropout,
                    layer_idx=layer_idx,
                    retrieval_config=self.retrieval_config,
                )
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")
            self.blocks.append(
                TransformerBlock(d_model, n_heads, d_ff, dropout, attn, layer_idx=layer_idx)
            )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.num_digit_classes = num_digit_classes
        self.digit_head = (
            nn.Linear(d_model, num_digit_classes) if num_digit_classes > 0 else None
        )
        self.pointer_mlp_head: MultiLayerPointerMLPHead | None = None
        if output_head == "pointer_mlp":
            mlp_hidden = max(pointer_mlp_hidden, max_seq_len + 1)
            self.pointer_mlp_head = MultiLayerPointerMLPHead(
                max_seq_len=max_seq_len,
                mlp_hidden=mlp_hidden,
                num_slot_outputs=num_pointer_slots,
                d_model=d_model,
            )
        self.pool_mlp_token_head: QueryPoolMLPTokenHead | None = None
        if output_head == "pool_mlp_token":
            self.pool_mlp_token_head = QueryPoolMLPTokenHead(
                max_seq_len=max_seq_len,
                d_model=d_model,
                vocab_size=vocab_size,
                pool_positions=pool_mlp_positions,
                n_layers=n_layers,
            )

    def get_router_parameters(self):
        """Return router params for freezing or separate optimizer."""
        if self.router is None:
            return []
        return list(self.router.parameters())

    def get_address_parameters(self):
        """Return learned-address params (PerLayerAddressBook or per-layer modules)."""
        if self.address_book is not None:
            return list(self.address_book.parameters())
        params = []
        for block in self.blocks:
            attn = block.attn
            if hasattr(attn, "address"):
                params.extend(attn.address.parameters())
        return params

    def freeze_router(self) -> None:
        for p in self.get_router_parameters():
            p.requires_grad = False

    def unfreeze_router(self) -> None:
        for p in self.get_router_parameters():
            p.requires_grad = True

    def freeze_addresses(self) -> None:
        for p in self.get_address_parameters():
            p.requires_grad = False

    def unfreeze_addresses(self) -> None:
        for p in self.get_address_parameters():
            p.requires_grad = True

    def compute_pointer_logits(
        self,
        hidden: torch.Tensor,
        question_index: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Query-conditioned pointer scores: scaled dot(h_question, h_i) for i < question_index."""
        B, T, _ = hidden.shape
        batch_idx = torch.arange(B, device=hidden.device)
        h_q = hidden[batch_idx, question_index]
        scores = torch.einsum("bd,btd->bt", h_q, hidden)
        scores = scores * (1.0 / math.sqrt(self.d_model))
        pos = torch.arange(T, device=hidden.device).unsqueeze(0)
        valid = pos < question_index.unsqueeze(1)
        if attn_mask is not None:
            valid = valid & attn_mask.bool()
        return scores.masked_fill(~valid, float("-inf"))

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        question_index: Optional[torch.Tensor] = None,
        pointer_target_index: Optional[torch.Tensor] = None,
        pointer_target_slot: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        return_attentions: bool = False,
        return_per_head_attention: bool = False,
        return_pre_attention_hidden: bool = False,
        target_layer: Optional[int] = None,
        stop_after_layer: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape
        if attn_mask is None:
            attn_mask = input_ids != self.pad_token_id

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        hidden_states = []
        pre_attention_hidden = []
        attentions = []
        layer_hidden_states: list[torch.Tensor] = []
        collect_pointer_layers = self.output_head in ("pointer_mlp", "pool_mlp_token")

        for layer_idx, block in enumerate(self.blocks):
            if target_layer is not None and layer_idx != target_layer:
                x = block(x, attn_mask=attn_mask)
                if return_hidden_states:
                    hidden_states.append(x)
                continue

            x_norm = block.ln1(x)
            if return_pre_attention_hidden:
                pre_attention_hidden.append(x_norm)

            if return_attentions:
                x, attn_w = block(
                    x,
                    attn_mask=attn_mask,
                    return_attention=True,
                    return_per_head=return_per_head_attention,
                )
                attentions.append(attn_w)
            else:
                x = block(x, attn_mask=attn_mask)
            if collect_pointer_layers:
                layer_hidden_states.append(x)
            if return_hidden_states:
                hidden_states.append(x)

            if stop_after_layer is not None and layer_idx >= stop_after_layer:
                break

        x = self.ln_f(x)
        logits = self.lm_head(x)

        output: dict[str, torch.Tensor] = {"logits": logits}

        uses_pointer = self.output_head in ("pointer_index", "pointer_mlp")
        if self.output_head == "pool_mlp_token":
            if question_index is None:
                question_index = torch.full(
                    (B,),
                    T - 1,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            assert self.pool_mlp_token_head is not None
            normalized_layers = [self.ln_f(h) for h in layer_hidden_states]
            output["token_logits"] = self.pool_mlp_token_head(normalized_layers, question_index)
        elif uses_pointer:
            if question_index is None:
                question_index = torch.full(
                    (B,),
                    T - 1,
                    device=input_ids.device,
                    dtype=torch.long,
                )
            if self.output_head == "pointer_mlp":
                assert self.pointer_mlp_head is not None
                pointer_logits = self.pointer_mlp_head(
                    layer_hidden_states,
                    question_index,
                    mode=self.pointer_target_mode,
                )
                if self.pointer_target_mode == "full_sequence":
                    pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
                    valid = pos < question_index.unsqueeze(1)
                    if attn_mask is not None:
                        valid = valid & attn_mask.bool()
                    pointer_logits = pointer_logits.masked_fill(~valid, float("-inf"))
            else:
                pointer_logits = self.compute_pointer_logits(x, question_index, attn_mask)
            output["pointer_logits"] = pointer_logits

            pointer_target = None
            if self.output_head == "pointer_mlp" and self.pointer_target_mode == "value_slots":
                pointer_target = pointer_target_slot
            else:
                pointer_target = pointer_target_index

            if pointer_target is not None:
                pointer_loss = F.cross_entropy(pointer_logits, pointer_target)
                output["loss"] = pointer_loss
                output["pointer_accuracy"] = (
                    pointer_logits.argmax(dim=-1) == pointer_target
                ).float().mean()
            else:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attn_mask[:, 1:].contiguous()
                per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(B, T - 1)
                output["loss"] = (per_token * shift_mask.float()).sum() / shift_mask.float().sum().clamp(min=1)
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attn_mask[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            loss = loss.view(B, T - 1)
            loss = (loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp(min=1)
            output["loss"] = loss
        if self.digit_head is not None and labels is not None:
            mask = attn_mask.float() if attn_mask is not None else torch.ones(B, T, device=x.device)
            pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            digit_logits = self.digit_head(pooled)
            digit_loss = F.cross_entropy(digit_logits, labels)
            output["digit_loss"] = digit_loss
            output["digit_accuracy"] = (digit_logits.argmax(dim=-1) == labels).float().mean()
        if return_hidden_states:
            output["hidden_states"] = hidden_states
        if return_pre_attention_hidden:
            output["pre_attention_hidden"] = pre_attention_hidden
        if return_attentions:
            output["attentions"] = attentions
        return output

    @torch.inference_mode()
    def collect_attention_data(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        per_head: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Collect hidden states and attention weights from a specific layer."""
        layer = layer_idx if layer_idx >= 0 else self.n_layers + layer_idx
        out = self.forward(
            input_ids,
            attn_mask=attn_mask,
            return_hidden_states=True,
            return_pre_attention_hidden=True,
            return_attentions=True,
            return_per_head_attention=per_head,
            target_layer=None,
            stop_after_layer=layer,
        )
        attn = out["attentions"][layer]
        if per_head and attn.dim() == 4:
            pass
        elif attn.dim() == 4:
            attn = attn.mean(dim=1)
        # Router trains on ln1(x) — same input RoutingSparseAttention uses at inference
        return {
            "hidden_states": out["pre_attention_hidden"][layer],
            "attention": attn,
            "layer_idx": layer,
            "per_head": per_head,
        }
