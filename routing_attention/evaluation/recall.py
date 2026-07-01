"""Recall@K and distance-stratified recall metrics for routing evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tqdm import tqdm

from routing_attention.models.learned_address import PerLayerAddressBook
from routing_attention.models.router import PerLayerRouter

_CAUSAL_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    key = (seq_len, device)
    if key not in _CAUSAL_CACHE:
        _CAUSAL_CACHE[key] = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return _CAUSAL_CACHE[key]


def resolve_max_eval_tokens(seq_len: int, max_eval_tokens: int = 0, dry_run: bool = False) -> int:
    """Pick how many token positions to score (0 = auto from seq_len)."""
    if max_eval_tokens > 0:
        return min(max_eval_tokens, seq_len)
    if dry_run and seq_len <= 784:
        # MNIST (784) and shorter sequences: score all tokens.
        return seq_len
    if dry_run:
        return min(128, seq_len)
    if seq_len > 4096:
        return 256
    if seq_len > 2048:
        return 384
    if seq_len > 1024:
        return 512
    return seq_len


def subsample_for_eval(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    max_samples: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample cached sequences for fast final evaluation (0 = use all)."""
    if max_samples <= 0 or hidden.shape[0] <= max_samples:
        return hidden, attention
    idx = torch.randperm(hidden.shape[0], device=hidden.device)[:max_samples]
    return hidden[idx], attention[idx]


def subsample_tokens_for_eval(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    max_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample token positions (causal submatrix) for long-sequence eval."""
    if max_tokens <= 0 or hidden.shape[1] <= max_tokens:
        return hidden, attention
    idx = torch.randperm(hidden.shape[1], device=hidden.device)[:max_tokens].sort().values
    hidden = hidden[:, idx]
    if attention.dim() == 4:
        attention = attention[:, :, idx][:, :, :, idx]
    else:
        attention = attention[:, idx][:, :, idx]
    return hidden, attention


def prepare_eval_tensors(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    device: torch.device,
    max_samples: int = 0,
    max_tokens: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample on source device, then transfer only the eval slice to GPU."""
    hidden, attention = subsample_for_eval(hidden, attention, max_samples)
    hidden, attention = subsample_tokens_for_eval(hidden, attention, max_tokens)
    non_blocking = device.type == "cuda"
    return (
        hidden.to(device, non_blocking=non_blocking),
        attention.to(device, non_blocking=non_blocking),
    )


def _overall_recall_from_topk(
    true_topk: torch.Tensor,
    pred_topk: torch.Tensor,
    attn_masked: torch.Tensor,
    k_eff: int,
) -> dict[str, float]:
    true_hits = (true_topk.unsqueeze(-1) == pred_topk.unsqueeze(-2)).any(dim=-1).float()
    recall_per_q = true_hits.sum(dim=-1) / k_eff
    pred_hits = (pred_topk.unsqueeze(-1) == true_topk.unsqueeze(-2)).any(dim=-1).float()
    precision_per_q = pred_hits.sum(dim=-1) / k_eff
    valid = attn_masked.sum(dim=-1) > 0
    recall_per_q = recall_per_q[valid]
    precision_per_q = precision_per_q[valid]
    return {
        f"recall@{k_eff}": recall_per_q.mean().item() if recall_per_q.numel() else 0.0,
        f"precision@{k_eff}": precision_per_q.mean().item() if precision_per_q.numel() else 0.0,
        "num_queries": int(recall_per_q.numel()),
        "mean_rank_true_neighbors": 0.0,
    }


@torch.inference_mode()
def compute_recall_at_k(
    routing: torch.Tensor,
    attention: torch.Tensor,
    k: int = 32,
    causal: bool = True,
    similarity: torch.Tensor | None = None,
    include_mean_rank: bool = False,
) -> dict[str, float]:
    """Measure Recall@K with fully vectorized GPU ops."""
    del include_mean_rank
    _, T, _ = attention.shape if attention.dim() == 3 else (attention.shape[0], attention.shape[-1], 0)
    k_eff = min(k, T)

    if similarity is not None:
        sim = similarity
    elif routing.dim() == 4:
        sim = torch.einsum("bthq,bshq->bts", routing, routing) / max(routing.shape[2], 1)
    else:
        sim = torch.matmul(routing, routing.transpose(-2, -1))

    if causal:
        causal_mask = _causal_mask(T, sim.device)
        attn_masked = attention * causal_mask.unsqueeze(0)
        sim = sim.masked_fill(causal_mask.unsqueeze(0) == 0, float("-inf"))
    else:
        attn_masked = attention

    _, true_topk = torch.topk(attn_masked, k=k_eff, dim=-1)
    _, pred_topk = torch.topk(sim, k=k_eff, dim=-1)
    return _overall_recall_from_topk(true_topk, pred_topk, attn_masked, k_eff)


@torch.inference_mode()
def compute_recall_from_router(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    router: nn.Module,
    layer_idx: int | None = None,
    k: int = 32,
    max_samples: int = 0,
    max_tokens: int = 0,
    include_mean_rank: bool = False,
) -> dict[str, float]:
    """Evaluate recall using router retrieval_scores (single forward pass)."""
    del include_mean_rank
    device = next(router.parameters()).device
    hidden, attention = prepare_eval_tensors(hidden, attention, device, max_samples, max_tokens)
    if attention.dim() == 4:
        attention = attention.mean(dim=1)

    if isinstance(router, (PerLayerRouter, PerLayerAddressBook)):
        assert layer_idx is not None
        sim = router.retrieval_scores(hidden, layer_idx)
    elif hasattr(router, "retrieval_scores"):
        sim = router.retrieval_scores(hidden)
    else:
        r = router(hidden)
        sim = torch.matmul(r, r.transpose(-2, -1))

    return compute_recall_at_k(
        routing=hidden,
        attention=attention,
        k=k,
        similarity=sim,
    )


@torch.inference_mode()
def compute_recall_by_distance(
    routing: torch.Tensor,
    attention: torch.Tensor,
    k: int = 32,
    distance_buckets: list[int] | None = None,
    similarity: torch.Tensor | None = None,
    max_samples: int = 0,
    max_tokens: int = 0,
) -> dict[str, Any]:
    """Stratify recall by token distance |i - j|."""
    del distance_buckets
    device = routing.device
    routing, attention = prepare_eval_tensors(routing, attention, device, max_samples, max_tokens)
    if attention.dim() == 4:
        attention = attention.mean(dim=1)

    _, T, _ = routing.shape if routing.dim() == 3 else (attention.shape[0], attention.shape[-1], 0)
    k_eff = min(k, T)

    if similarity is None:
        sim = torch.matmul(routing, routing.transpose(-2, -1))
    else:
        sim = similarity

    causal_mask = _causal_mask(T, attention.device)
    attn_masked = attention * causal_mask.unsqueeze(0)
    sim_masked = sim.masked_fill(causal_mask.unsqueeze(0) == 0, float("-inf"))

    _, true_topk = torch.topk(attn_masked, k=k_eff, dim=-1)
    _, pred_topk = torch.topk(sim_masked, k=k_eff, dim=-1)

    query_pos = torch.arange(T, device=attention.device).view(1, T, 1)
    dist_true = (query_pos - true_topk).abs()
    in_pred = (true_topk.unsqueeze(-1) == pred_topk.unsqueeze(-2)).any(dim=-1)

    bin_edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 10000]
    per_bin = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (dist_true > lo) & (dist_true <= hi)
        hits = (in_bin & in_pred).sum().item()
        total = in_bin.sum().item()
        per_bin.append({
            "distance_min": lo,
            "distance_max": hi,
            "recall": hits / max(total, 1),
            "hits": hits,
            "total": total,
        })

    return {
        "buckets": {},
        "per_bin": per_bin,
        "overall": _overall_recall_from_topk(true_topk, pred_topk, attn_masked, k_eff),
    }


@torch.inference_mode()
def compute_recall_by_distance_from_router(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    router: nn.Module,
    layer_idx: int | None = None,
    k: int = 32,
    max_samples: int = 0,
    max_tokens: int = 0,
) -> dict[str, Any]:
    """Distance-stratified recall; router forward runs only on the eval subsample."""
    device = next(router.parameters()).device
    hidden, attention = prepare_eval_tensors(hidden, attention, device, max_samples, max_tokens)
    if attention.dim() == 4:
        attention = attention.mean(dim=1)

    if isinstance(router, (PerLayerRouter, PerLayerAddressBook)):
        assert layer_idx is not None
        sim = router.retrieval_scores(hidden, layer_idx)
    elif hasattr(router, "retrieval_scores"):
        sim = router.retrieval_scores(hidden)
    else:
        r = router(hidden)
        sim = torch.matmul(r, r.transpose(-2, -1))

    return compute_recall_by_distance(
        hidden,
        attention,
        k=k,
        similarity=sim,
        max_samples=0,
        max_tokens=0,
    )


@torch.inference_mode()
def compute_random_routing_baseline(
    cache: dict,
    router: nn.Module,
    device: torch.device,
    recall_k: int,
    max_eval_samples: int = 0,
    max_eval_tokens: int = 0,
    per_layer: bool = True,
    n_layers: int = 1,
) -> dict[str, Any]:
    """Random-unit-vector routing recall — estimates chance overlap at this eval budget."""
    if per_layer and "layers" in cache:
        per_layer_recall = {}
        for li in range(n_layers):
            layer_data = cache["layers"][li]
            hidden, attention = prepare_eval_tensors(
                layer_data["hidden_states"],
                layer_data["attention"],
                device,
                max_eval_samples,
                max_eval_tokens,
            )
            if attention.dim() == 4:
                attention = attention.mean(dim=1)
            rand = torch.randn_like(hidden)
            rand = rand / rand.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            per_layer_recall[f"layer_{li}"] = compute_recall_at_k(rand, attention, k=recall_k)
        seq_len = cache["layers"][0]["hidden_states"].shape[1]
        key = f"recall@{min(recall_k, seq_len)}"
        return {
            "per_layer": per_layer_recall,
            "mean_recall": sum(v.get(key, 0) for v in per_layer_recall.values()) / max(len(per_layer_recall), 1),
        }

    hidden = cache["hidden_states"]
    attention = cache["attention"]
    hidden, attention = prepare_eval_tensors(hidden, attention, device, max_eval_samples, max_eval_tokens)
    if attention.dim() == 4:
        attention = attention.mean(dim=1)
    rand = torch.randn_like(hidden)
    rand = rand / rand.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    result = compute_recall_at_k(rand, attention, k=recall_k)
    return {"mean_recall": result.get(f"recall@{min(recall_k, hidden.shape[1])}", 0), "overall": result}


@torch.inference_mode()
def compute_key_vector_recall_from_hidden(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    attn_module: nn.Module,
    k: int = 32,
) -> dict[str, float]:
    """Recall@K using dense attention's Q/K projections (no RouterMLP)."""
    if not hasattr(attn_module, "q_proj"):
        raise TypeError("attn_module must be DenseAttention with q_proj/k_proj")
    B, T, D = hidden.shape
    n_heads = attn_module.n_heads
    head_dim = attn_module.head_dim
    q = attn_module.q_proj(hidden).view(B, T, n_heads, head_dim).transpose(1, 2)
    k_proj = attn_module.k_proj(hidden).view(B, T, n_heads, head_dim).transpose(1, 2)
    q_mean = q.mean(dim=1)
    k_mean = k_proj.mean(dim=1)
    sim = torch.matmul(q_mean, k_mean.transpose(-2, -1))
    if attention.dim() == 4:
        attention = attention.mean(dim=1)
    return compute_recall_at_k(hidden, attention, k=k, similarity=sim)


@torch.inference_mode()
def evaluate_learned_address_recall_from_cache(
    address_book: nn.Module,
    cache_path: Path | str,
    device: torch.device,
    recall_k: int,
    n_layers: int,
    max_eval_samples: int = 0,
    max_eval_tokens: int = 0,
    show_progress: bool = True,
    layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Per-layer Recall@K using learned address projections (meat unused for retrieval)."""
    from routing_attention.data.chunked_cache import ChunkedAttentionCache, is_chunked_cache

    path = Path(cache_path)
    layer_indices = layer_indices if layer_indices is not None else list(range(n_layers))
    per_layer = {}
    layer_iter = layer_indices
    if show_progress:
        layer_iter = tqdm(layer_indices, desc="Learned-address Recall@K", unit="layer")

    seq_len = None
    for li in layer_iter:
        if is_chunked_cache(path):
            hidden, attention = ChunkedAttentionCache(path).load_layer_tensors(
                li, device, max_samples=max_eval_samples,
            )
        else:
            data = torch.load(path, map_location="cpu", weights_only=False)
            layer = data["layers"][li] if "layers" in data else data
            hidden = layer["hidden_states"][:max_eval_samples or None].to(device)
            attention = layer["attention"][:max_eval_samples or None].to(device)

        if max_eval_tokens > 0 and hidden.shape[1] > max_eval_tokens:
            hidden = hidden[:, :max_eval_tokens]
            if attention.dim() == 3:
                attention = attention[:, :max_eval_tokens, :max_eval_tokens]
            else:
                attention = attention[:, :, :max_eval_tokens, :max_eval_tokens]

        seq_len = hidden.shape[1]
        addr_mod = (
            address_book.get_address(li)
            if isinstance(address_book, PerLayerAddressBook)
            else address_book
        )
        per_layer[f"layer_{li}"] = compute_recall_from_router(
            hidden, attention, addr_mod, layer_idx=li, k=recall_k,
        )

    key = f"recall@{min(recall_k, seq_len or recall_k)}"
    return {
        "per_layer": per_layer,
        "mean_recall": sum(v.get(key, 0) for v in per_layer.values()) / max(len(per_layer), 1),
        "eval_mode": "learned_address",
    }


@torch.inference_mode()
def evaluate_key_vector_recall_from_cache(
    model: nn.Module,
    cache_path: Path | str,
    device: torch.device,
    recall_k: int,
    n_layers: int,
    max_eval_samples: int = 0,
    max_eval_tokens: int = 0,
    show_progress: bool = True,
    layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Per-layer Recall@K from frozen transformer Q/K projections on cached hidden states."""
    from routing_attention.data.chunked_cache import ChunkedAttentionCache, is_chunked_cache

    path = Path(cache_path)
    layer_indices = layer_indices if layer_indices is not None else list(range(n_layers))
    per_layer = {}
    layer_iter = layer_indices
    if show_progress:
        layer_iter = tqdm(layer_indices, desc="Key-vector Recall@K", unit="layer")

    seq_len = None
    for li in layer_iter:
        if is_chunked_cache(path):
            hidden, attention = ChunkedAttentionCache(path).load_layer_tensors(
                li, device, max_samples=max_eval_samples,
            )
        else:
            data = torch.load(path, map_location="cpu", weights_only=False)
            layer = data["layers"][li] if "layers" in data else data
            hidden = layer["hidden_states"][:max_eval_samples or None].to(device)
            attention = layer["attention"][:max_eval_samples or None].to(device)

        if max_eval_tokens > 0 and hidden.shape[1] > max_eval_tokens:
            hidden = hidden[:, :max_eval_tokens]
            if attention.dim() == 3:
                attention = attention[:, :max_eval_tokens, :max_eval_tokens]
            else:
                attention = attention[:, :, :max_eval_tokens, :max_eval_tokens]

        seq_len = hidden.shape[1]
        attn_mod = model.blocks[li].attn
        per_layer[f"layer_{li}"] = compute_key_vector_recall_from_hidden(
            hidden, attention, attn_mod, k=recall_k,
        )

    key = f"recall@{min(recall_k, seq_len or recall_k)}"
    return {
        "per_layer": per_layer,
        "mean_recall": sum(v.get(key, 0) for v in per_layer.values()) / max(len(per_layer), 1),
        "eval_mode": "key_vector",
    }


def _normalize_eval_layer_indices(eval_layers: list[int], n_layers: int) -> list[int]:
    out = []
    for idx in eval_layers:
        li = idx if idx >= 0 else n_layers + idx
        if 0 <= li < n_layers:
            out.append(li)
    return out or list(range(n_layers))


def filter_recall_metrics_by_layers(recall_metrics: dict[str, Any], layer_indices: list[int]) -> dict[str, Any]:
    """Restrict per-layer recall dict and recompute mean_recall."""
    per_layer = recall_metrics.get("per_layer", {})
    if not per_layer:
        return recall_metrics
    selected = {f"layer_{li}": per_layer[f"layer_{li}"] for li in layer_indices if f"layer_{li}" in per_layer}
    if not selected:
        return recall_metrics
    keys = [k for k in next(iter(selected.values())) if k.startswith("recall@")]
    mean_recall = 0.0
    if keys:
        mean_recall = sum(v.get(keys[0], 0) for v in selected.values()) / len(selected)
    out = {**recall_metrics, "per_layer": selected, "mean_recall": mean_recall, "eval_layers": layer_indices}
    rb = recall_metrics.get("random_baseline", {})
    if isinstance(rb, dict) and "per_layer" in rb:
        rb_sel = {f"layer_{li}": rb["per_layer"][f"layer_{li}"] for li in layer_indices if f"layer_{li}" in rb["per_layer"]}
        if rb_sel and keys:
            rb_mean = sum(v.get(keys[0], 0) for v in rb_sel.values()) / len(rb_sel)
            out["random_baseline"] = {**rb, "per_layer": rb_sel, "mean_recall": rb_mean}
            out["recall_above_random"] = mean_recall - rb_mean
    return out


def evaluate_per_layer_recall(
    router: nn.Module,
    cache: dict,
    device: torch.device,
    recall_k: int,
    n_layers: int,
    max_eval_samples: int = 0,
    max_eval_tokens: int = 0,
    include_mean_rank: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run final Recall@K across layers with optional progress bar."""
    all_recall = {}
    layer_iter = range(n_layers)
    if show_progress:
        layer_iter = tqdm(layer_iter, desc="Recall@K per layer", unit="layer")

    router.eval()
    for li in layer_iter:
        layer_data = cache["layers"][li]
        all_recall[f"layer_{li}"] = compute_recall_from_router(
            layer_data["hidden_states"],
            layer_data["attention"],
            router,
            layer_idx=li,
            k=recall_k,
            max_samples=max_eval_samples,
            max_tokens=max_eval_tokens,
            include_mean_rank=include_mean_rank,
        )

    seq_len = cache["layers"][0]["hidden_states"].shape[1]
    key = f"recall@{min(recall_k, seq_len)}"
    return {
        "per_layer": all_recall,
        "mean_recall": sum(v.get(key, 0) for v in all_recall.values()) / max(len(all_recall), 1),
    }
