"""Production attention backend manifest for Experiment 7 fair comparisons."""

from __future__ import annotations

from typing import Any

# Maps attention_type → production kernel used in the suite.
EXP7_PRODUCTION_BACKENDS: dict[str, dict[str, str]] = {
    "dense": {
        "kernel": "pytorch_sdpa_flash",
        "package": "torch.nn.functional.scaled_dot_product_attention",
        "notes": "Same fused path as dense_flash (no naive T×T materialization).",
    },
    "dense_flash": {
        "kernel": "pytorch_sdpa_flash",
        "package": "torch.nn.functional.scaled_dot_product_attention",
        "notes": "Flash/mem-efficient SDPA when available.",
    },
    "linear": {
        "kernel": "fla_chunk_linear_attn",
        "package": "flash-linear-attention (fla.ops.linear_attn)",
        "notes": "Auto-selects chunk vs fused_chunk; chunk fastest at H=4 D=64.",
    },
    "local": {
        "kernel": "pytorch_flex_attention",
        "package": "torch.nn.attention.flex_attention (torch.compile)",
        "notes": "Causal sliding-window softmax; no Python timestep loops.",
    },
    "routing": {
        "kernel": "sparse_topk_retrieval",
        "package": "routing_attention (RoutingRetriever + sparse meat)",
        "notes": "Top-k only; use_fused_sparse=true when Triton available.",
    },
    "key_vector": {
        "kernel": "sparse_topk_key_vectors",
        "package": "routing_attention (KeyVectorSparseAttention)",
        "notes": "Same sparse meat path as routing.",
    },
    "learned_address": {
        "kernel": "sparse_topk_learned_address",
        "package": "routing_attention (LearnedAddressSparseAttention)",
        "notes": "Same sparse meat path as routing.",
    },
}


_EXP7_VARIANT_ATTN: dict[str, str] = {
    "dense": "dense",
    "dense_flash": "dense_flash",
    "linear": "linear",
    "local_window64": "local",
    "local_window256": "local",
    "routing": "routing",
    "routing_asymmetric": "routing",
    "key_vector_k32": "key_vector",
    "learned_address_k32": "learned_address",
}


def variant_attention_type(variant: str) -> str:
    return _EXP7_VARIANT_ATTN.get(variant, variant)


def production_manifest_for_variants(variants: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for v in variants:
        attn = variant_attention_type(v)
        out[v] = {
            "attention_type": attn,
            **EXP7_PRODUCTION_BACKENDS.get(attn, {"kernel": "unknown"}),
        }
    return out


def assert_production_backends_available(variants: list[str] | None = None) -> None:
    from routing_attention.models.fast_attention import backend_status, require_fla, require_flex_attention

    status = backend_status()
    active = variants or list(_EXP7_VARIANT_ATTN.keys())
    attn_types = {variant_attention_type(v) for v in active}

    if attn_types & {"dense", "dense_flash"} and not status.get("sdpa_flash"):
        raise RuntimeError("PyTorch Flash SDPA not enabled — enable cuda flash SDP for dense baseline")
    if "linear" in attn_types:
        require_fla()
    if "local" in attn_types:
        require_flex_attention()
