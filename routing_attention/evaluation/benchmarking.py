"""Inference time and memory benchmarking for attention mechanisms."""

from __future__ import annotations

import time
from typing import Any

import psutil
import torch
import torch.nn as nn
from tqdm import tqdm

from routing_attention.retrieval.index import (
    RetrievalConfig,
    RoutingRetriever,
    numpy_bridge_available,
    retrieval_config_from_dict,
)


def measure_memory_usage() -> dict[str, float]:
    process = psutil.Process()
    mem = process.memory_info()
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "vms_mb": mem.vms / (1024 * 1024),
    }


@torch.no_grad()
def benchmark_routing_retrieval(
    seq_len: int,
    batch_size: int = 1,
    routing_dim: int = 32,
    top_k: int = 32,
    device: torch.device | None = None,
    retrieval_cfg: dict | None = None,
    num_runs: int = 20,
) -> dict[str, Any]:
    """Benchmark vector-search retrieval in isolation (ms per layer-call)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = retrieval_config_from_dict(retrieval_cfg or {})
    cfg.top_k = top_k
    cfg.max_seq_len = max(cfg.max_seq_len, seq_len)

    retriever = RoutingRetriever(cfg).to(device)
    q = torch.randn(batch_size, seq_len, routing_dim, device=device)
    k = torch.randn(batch_size, seq_len, routing_dim, device=device)
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    times = retriever.benchmark(q, k, top_k=top_k, num_runs=num_runs)
    resolved = retriever.resolve_method(seq_len)
    return {
        "seq_len": seq_len,
        "resolved_method": resolved,
        "retrieval_ms_per_call": times,
        "routing_dim": routing_dim,
        "top_k": top_k,
    }


@torch.no_grad()
def benchmark_attention(
    model: nn.Module,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    vocab_size: int = 256,
    num_warmup: int = 3,
    num_runs: int = 10,
    retrieval_cfg: dict | None = None,
) -> dict[str, Any]:
    """Benchmark full forward pass + optional routing retrieval breakdown."""
    model.eval()
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
    attn_mask = torch.ones_like(input_ids)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    for _ in tqdm(range(num_warmup), desc="Benchmark warmup", leave=False):
        _ = model(input_ids, attn_mask=attn_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()

    mem_before = measure_memory_usage()
    start = time.perf_counter()
    for _ in tqdm(range(num_runs), desc="Benchmark forward", leave=False):
        _ = model(input_ids, attn_mask=attn_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = batch_size * seq_len * num_runs
    latency_ms = (elapsed / num_runs) * 1000

    result: dict[str, Any] = {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "numpy_bridge_ok": numpy_bridge_available(),
        "latency_ms": latency_ms,
        "throughput_tokens_per_sec": tokens / elapsed,
        "num_runs": num_runs,
        "cpu_rss_mb_before": mem_before["rss_mb"],
        "cpu_rss_mb_after": measure_memory_usage()["rss_mb"],
    }

    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_peak_memory_mb"] = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        result["gpu_current_memory_mb"] = torch.cuda.memory_allocated(device) / (1024 * 1024)

    top_k = getattr(model, "routing_top_k", 32)
    dense_edges = seq_len * seq_len
    sparse_edges = seq_len * top_k
    n_layers = getattr(model, "n_layers", 1)
    result["dense_edges_per_layer"] = dense_edges
    result["sparse_edges_per_layer"] = sparse_edges
    result["edge_compression_ratio"] = dense_edges / max(sparse_edges, 1)
    result["total_sparse_edges_all_layers"] = sparse_edges * n_layers

    # Isolated retrieval benchmark for sparse attention variants
    attn_type = getattr(model, "attention_type", None)
    search_dim = None
    include_retrieval_bench = False
    if retrieval_cfg is not None:
        if attn_type == "routing":
            search_dim = 32
            include_retrieval_bench = True
        elif attn_type == "learned_address":
            search_dim = 32
            include_retrieval_bench = True
        elif attn_type == "key_vector" and (
            retrieval_cfg.get("apply_to_key_vector", False)
            or retrieval_cfg.get("use_fused_sparse", False)
        ):
            d_model = getattr(model, "d_model", 256)
            n_heads = getattr(model, "n_heads", 4)
            search_dim = d_model // n_heads
            include_retrieval_bench = True

    if include_retrieval_bench and search_dim is not None:
        rbench = benchmark_routing_retrieval(
            seq_len=seq_len,
            batch_size=batch_size,
            routing_dim=search_dim,
            top_k=top_k,
            device=device,
            retrieval_cfg=retrieval_cfg,
            num_runs=num_runs,
        )
        result["routing_retrieval"] = rbench
        result["retrieval_search_dim"] = search_dim
        result["key_vector_fast_retrieval"] = attn_type == "key_vector"
        resolved = rbench["resolved_method"]
        per_layer_ms = rbench["retrieval_ms_per_call"].get(resolved, 0)
        result["routing_retrieval_ms_per_layer"] = per_layer_ms
        result["routing_retrieval_ms_all_layers"] = per_layer_ms * n_layers
        result["estimated_retrieval_fraction"] = min(
            1.0,
            (per_layer_ms * n_layers) / max(latency_ms, 1e-6),
        )

    return result
