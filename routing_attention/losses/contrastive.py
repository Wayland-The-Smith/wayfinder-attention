"""Contrastive / InfoNCE losses for routing neighborhood prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for routing vector neighborhood learning.

    For each query token i, positives are tokens in top-k attention neighbors,
    negatives are sampled from non-neighbor tokens.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        routing: torch.Tensor,
        positive_mask: torch.Tensor,
        num_negatives: int = 64,
        key_routing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            routing: (B, T, R) L2-normalized routing vectors
            positive_mask: (B, T, T) bool, True if j is positive neighbor of i
            num_negatives: negatives sampled per query

        Returns:
            scalar loss
        """
        B, T = routing.shape[:2]
        key = key_routing if key_routing is not None else routing
        if routing.dim() == 4:
            sim = torch.einsum("bthq,bshq->bts", routing, key) / max(routing.shape[2], 1) / self.temperature
        else:
            sim = torch.matmul(routing, key.transpose(-2, -1)) / self.temperature

        losses = []
        for b in range(B):
            for i in range(T):
                pos_idx = positive_mask[b, i].nonzero(as_tuple=True)[0]
                if pos_idx.numel() == 0:
                    continue

                # Causal: only j <= i
                valid_keys = torch.arange(T, device=routing.device)
                valid_keys = valid_keys[valid_keys <= i]

                neg_candidates = valid_keys[~positive_mask[b, i, valid_keys]]
                if neg_candidates.numel() == 0:
                    neg_idx = valid_keys
                else:
                    n_neg = min(num_negatives, neg_candidates.numel())
                    perm = torch.randperm(neg_candidates.numel(), device=routing.device)[:n_neg]
                    neg_idx = neg_candidates[perm]

                # All keys for this query: positives + negatives
                all_idx = torch.cat([pos_idx, neg_idx]).unique()
                logits = sim[b, i, all_idx]

                # Labels: first len(pos_idx) entries are positives
                pos_set = set(pos_idx.tolist())
                labels = torch.tensor(
                    [1 if all_idx[k].item() in pos_set else 0 for k in range(all_idx.numel())],
                    device=routing.device,
                    dtype=torch.float,
                )
                # Multi-positive InfoNCE: log(sum exp pos) - log(sum exp all)
                pos_logits = logits[labels == 1]
                if pos_logits.numel() == 0:
                    continue
                loss_i = -torch.logsumexp(pos_logits, dim=0) + torch.logsumexp(logits, dim=0)
                losses.append(loss_i)

        if not losses:
            return torch.tensor(0.0, device=routing.device, requires_grad=True)
        return torch.stack(losses).mean()


def contrastive_routing_loss(
    routing: torch.Tensor,
    attention: torch.Tensor,
    top_k: int = 32,
    temperature: float = 0.07,
    num_negatives: int = 64,
) -> torch.Tensor:
    """
    Efficient batched contrastive loss using top-k attention neighbors as positives.

    Args:
        routing: (B, T, R) normalized
        attention: (B, T, T) attention weights (post-softmax)
        top_k: number of positive neighbors per query
    """
    B, T, _ = attention.shape

    # Build positive mask from top-k attention neighbors (causal)
    attn_masked = attention.clone()
    causal = torch.tril(torch.ones(T, T, device=attention.device))
    attn_masked = attn_masked * causal.unsqueeze(0)

    k_eff = min(top_k, T)
    _, top_idx = torch.topk(attn_masked, k=k_eff, dim=-1)
    positive_mask = torch.zeros(B, T, T, dtype=torch.bool, device=attention.device)
    positive_mask.scatter_(2, top_idx, True)

    # Exclude self-attention from positives (optional - keep self as it's always attended)
    criterion = InfoNCELoss(temperature=temperature)
    return criterion(routing, positive_mask, num_negatives=num_negatives)


def sampled_infonce_loss(
    routing: torch.Tensor,
    attention: torch.Tensor,
    top_k: int = 32,
    temperature: float = 0.07,
    num_negatives: int = 64,
    key_routing: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    InfoNCE with sampled negatives — vectorized via batched_infonce_loss.

    Uses full causal denominator (efficient GPU kernel) instead of per-token Python loops.
    """
    del num_negatives  # batched kernel uses all valid keys in denominator
    return batched_infonce_loss(
        routing, attention, top_k=top_k, temperature=temperature, key_routing=key_routing,
    )


_CAUSAL_MASK_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    key = (seq_len, device)
    cached = _CAUSAL_MASK_CACHE.get(key)
    if cached is None:
        cached = torch.tril(torch.ones(seq_len, seq_len, device=device))
        _CAUSAL_MASK_CACHE[key] = cached
    return cached


def _routing_similarity(
    query: torch.Tensor,
    key: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    key = key if key is not None else query
    if query.dim() == 4:
        sim = torch.einsum("bthq,bshq->bts", query, key) / max(query.shape[2], 1)
    else:
        sim = torch.matmul(query, key.transpose(-2, -1))
    return sim / temperature


def batched_infonce_loss(
    routing: torch.Tensor,
    attention: torch.Tensor,
    top_k: int = 32,
    temperature: float = 0.07,
    key_routing: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Vectorized InfoNCE: for each (b, i), positives are top-k attended keys.

    Uses all other valid keys as negatives in the denominator (efficient approximation).
    """
    B, T, _ = attention.shape
    sim = _routing_similarity(routing, key_routing, temperature)

    causal = _causal_mask(T, routing.device)
    attn_masked = attention * causal.unsqueeze(0)
    k_eff = min(top_k, T)
    _, top_idx = torch.topk(attn_masked, k=k_eff, dim=-1)

    positive_mask = torch.zeros(B, T, T, dtype=torch.bool, device=routing.device)
    positive_mask.scatter_(2, top_idx, True)

    sim = sim.masked_fill(causal.unsqueeze(0) == 0, float("-inf"))

    # For each query, log(sum exp(pos)) - log(sum exp(all valid))
    pos_sim = sim.masked_fill(~positive_mask, float("-inf"))
    log_pos = torch.logsumexp(pos_sim, dim=-1)  # (B, T)
    log_all = torch.logsumexp(sim, dim=-1)

    valid_queries = positive_mask.any(dim=-1)  # (B, T)
    loss = -(log_pos - log_all)
    loss = loss[valid_queries]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=routing.device, requires_grad=True)
    return loss.mean()
