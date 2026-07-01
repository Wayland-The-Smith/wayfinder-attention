"""Apply Experiment 7 routing-variant settings (freeze router, retriever capacity)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from routing_attention.retrieval.index import patch_model_retrievers


def apply_routing_variant_settings(
    model: nn.Module,
    var_config: dict[str, Any],
    attn_type: str,
    max_seq_len: int,
) -> dict[str, Any]:
    """
    Mirror Experiment 4 routing setup:
      - freeze router when configured (train meat + trunk only)
      - resize retriever buffers for long-context eval
    """
    info: dict[str, Any] = {"attn_type": attn_type}
    ra_cfg = var_config.get("routing_attention", {})

    if attn_type == "routing":
        freeze = ra_cfg.get("freeze_router", True)
        if freeze and hasattr(model, "freeze_router"):
            model.freeze_router()
        info["freeze_router"] = freeze
        n_routing = sum(
            1 for b in model.blocks if type(b.attn).__name__ == "RoutingSparseAttention"
        )
        info["routing_layers"] = n_routing
        if n_routing != model.n_layers:
            raise RuntimeError(
                f"Expected routing at all {model.n_layers} layers, got {n_routing}"
            )
        router_params = list(model.get_router_parameters()) if hasattr(model, "get_router_parameters") else []
        info["router_params_trainable"] = sum(p.requires_grad for p in router_params)

    if attn_type == "learned_address":
        freeze_addr = ra_cfg.get("freeze_addresses", ra_cfg.get("freeze_router", True))
        if freeze_addr and hasattr(model, "freeze_addresses"):
            model.freeze_addresses()
        elif hasattr(model, "unfreeze_addresses"):
            model.unfreeze_addresses()
        info["freeze_addresses"] = freeze_addr
        if hasattr(model, "get_address_parameters"):
            addr_params = list(model.get_address_parameters())
            info["address_params_trainable"] = sum(p.requires_grad for p in addr_params)
            info["address_params_total"] = len(addr_params)

    if attn_type in ("routing", "key_vector", "learned_address"):
        patch_model_retrievers(model, max_seq_len)
        info["retrievers_patched"] = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info["trainable_params"] = trainable
    info["total_params"] = total
    return info
