"""Losses that train routing vectors to factorize attention scores directly."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mse_attention_loss(
    routing: torch.Tensor,
    attention: torch.Tensor,
    causal: bool = True,
    router: nn.Module | None = None,
    key_routing: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Train routing similarity to approximate attention weights.

    Uses learnable affine dot calibration when router has use_affine_dot.
    """
    if router is not None and hasattr(router, "pairwise_scores"):
        pred = router.pairwise_scores(routing, key_routing)
    elif key_routing is not None:
        pred = torch.matmul(routing, key_routing.transpose(-2, -1))
    else:
        pred = torch.matmul(routing, routing.transpose(-2, -1))

    target = attention
    if causal:
        T = attention.shape[-1]
        mask = torch.tril(torch.ones(T, T, device=attention.device))
        pred = pred * mask.unsqueeze(0)
        target = target * mask.unsqueeze(0)
    return F.mse_loss(pred, target)


def kl_attention_loss(
    routing: torch.Tensor,
    attention: torch.Tensor,
    temperature: float = 1.0,
    causal: bool = True,
    router: nn.Module | None = None,
    key_routing: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence between softmax(routing_sim / temp) and attention distribution."""
    if router is not None and hasattr(router, "pairwise_scores"):
        logits = router.pairwise_scores(routing, key_routing) / temperature
    elif key_routing is not None:
        logits = torch.matmul(routing, key_routing.transpose(-2, -1)) / temperature
    else:
        logits = torch.matmul(routing, routing.transpose(-2, -1)) / temperature

    T = attention.shape[-1]
    causal_mask = torch.tril(torch.ones(T, T, device=attention.device))
    if causal:
        logits = logits.masked_fill(causal_mask.unsqueeze(0) == 0, float("-inf"))

    log_pred = F.log_softmax(logits, dim=-1)
    target = attention
    if causal:
        target = target * causal_mask.unsqueeze(0)
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    return F.kl_div(log_pred, target, reduction="batchmean")


def multi_scale_routing_loss(
    local_routing: torch.Tensor,
    global_routing: torch.Tensor,
    attention: torch.Tensor,
    top_k: int = 32,
    temperature: float = 0.07,
    loss_type: str = "infonce",
) -> torch.Tensor:
    """Combined loss for multi-scale routers (local + global each supervised)."""
    from routing_attention.losses.contrastive import batched_infonce_loss, sampled_infonce_loss

    if loss_type == "infonce_sampled":
        l_loss = sampled_infonce_loss(local_routing, attention, top_k=top_k, temperature=temperature)
        g_loss = sampled_infonce_loss(global_routing, attention, top_k=top_k, temperature=temperature)
    elif loss_type == "infonce":
        l_loss = batched_infonce_loss(local_routing, attention, top_k=top_k, temperature=temperature)
        g_loss = batched_infonce_loss(global_routing, attention, top_k=top_k, temperature=temperature)
    else:
        raise ValueError(f"multi_scale supports infonce losses, got {loss_type}")
    return 0.5 * (l_loss + g_loss)
