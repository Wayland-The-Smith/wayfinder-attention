#!/usr/bin/env python3
"""
Microbenchmark causal top-K retrieval (Layer 2).

Compares reference, brute_force, streaming, fused_causal (Triton), and optional
FAISS paths across sequence-length / routing-dim sweeps.

Usage:
  python scripts/benchmark_causal_topk.py
  python scripts/benchmark_causal_topk.py --seq-lens 512 2048 8192 --dims 32 64
  python scripts/benchmark_causal_topk.py --correctness-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from routing_attention.kernels.causal_topk import (
    causal_topk,
    causal_topk_available,
    causal_topk_reference,
)
from routing_attention.kernels._common import has_triton
from routing_attention.retrieval.index import (
    RoutingRetriever,
    RetrievalConfig,
    _HAS_FAISS,
    numpy_bridge_available,
)


def _sorted_neighbor_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Sorted dot-product scores for each (B,T) row's neighbor set."""
    B, T, K = indices.shape
    out = torch.empty(B, T, K, device=q.device, dtype=torch.float32)
    for b in range(B):
        for t in range(T):
            idx = indices[b, t].long().clamp(0, T - 1)
            scores = (q[b, t].float() * k[b, idx].float()).sum(-1)
            out[b, t] = scores.sort().values
    return out


def check_indices_equivalent(
    q: torch.Tensor,
    k: torch.Tensor,
    ref: torch.Tensor,
    test: torch.Tensor,
    atol: float = 1e-2,
) -> tuple[bool, float]:
    """True if test indices achieve the same sorted score multiset as reference."""
    ref_s = _sorted_neighbor_scores(q, k, ref)
    test_s = _sorted_neighbor_scores(q, k, test)
    ok = torch.allclose(ref_s, test_s, atol=atol, rtol=0)
    max_err = (ref_s - test_s).abs().max().item() if ref_s.numel() else 0.0
    return bool(ok), max_err


@torch.no_grad()
def time_method(
    q: torch.Tensor,
    k: torch.Tensor,
    top_k: int,
    method: str,
    warmup: int,
    runs: int,
) -> float:
    for _ in range(warmup):
        causal_topk(q, k, top_k, method=method)
    if q.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        causal_topk(q, k, top_k, method=method)
    if q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


@torch.no_grad()
def time_retriever_method(
    retriever: RoutingRetriever,
    q: torch.Tensor,
    k: torch.Tensor,
    top_k: int,
    method: str,
    warmup: int,
    runs: int,
) -> float:
    for _ in range(warmup):
        retriever(q, k, top_k=top_k, method=method)  # type: ignore[arg-type]
    if q.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        retriever(q, k, top_k=top_k, method=method)  # type: ignore[arg-type]
    if q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def run_correctness(
    device: torch.device,
    seq_lens: list[int],
    dims: list[int],
    top_k: int,
    batch_size: int,
) -> list[str]:
    errors: list[str] = []
    methods = ["brute_force", "streaming"]
    if causal_topk_available():
        methods.append("fused_causal")

    for T in seq_lens:
        for R in dims:
            q = torch.randn(batch_size, T, R, device=device)
            k = torch.randn(batch_size, T, R, device=device)
            ref = causal_topk_reference(q, k, top_k)
            for method in methods:
                got = causal_topk(q, k, top_k, method=method)
                ok, err = check_indices_equivalent(q, k, ref, got)
                label = f"T={T} R={R} {method}"
                if not ok:
                    errors.append(f"{label}: max score err {err:.4g}")
                else:
                    print(f"  OK {label}")
    return errors


def run_benchmark(
    device: torch.device,
    seq_lens: list[int],
    dims: list[int],
    top_k: int,
    batch_size: int,
    warmup: int,
    runs: int,
    include_faiss: bool,
) -> None:
    kernel_methods = ["reference", "brute_force", "streaming"]
    if causal_topk_available():
        kernel_methods.append("fused_causal")

    print(f"\nDevice: {device}  triton={has_triton()}  fused_available={causal_topk_available()}")
    print(f"top_k={top_k}  batch={batch_size}  warmup={warmup}  runs={runs}\n")

    header = f"{'T':>6} {'R':>4} {'method':>14} {'ms':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for T in seq_lens:
        for R in dims:
            q = torch.randn(batch_size, T, R, device=device)
            k = torch.randn(batch_size, T, R, device=device)
            ref_ms = None

            for method in kernel_methods:
                if method == "reference":
                    m = "brute_force"
                    fn = lambda: causal_topk_reference(q, k, top_k)  # noqa: E731
                    for _ in range(warmup):
                        fn()
                    if q.is_cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(runs):
                        fn()
                    if q.is_cuda:
                        torch.cuda.synchronize()
                    ms = (time.perf_counter() - t0) / runs * 1000.0
                else:
                    ms = time_method(q, k, top_k, method, warmup, runs)

                if method == "reference":
                    ref_ms = ms
                speedup = ref_ms / ms if ref_ms and method != "reference" else float("nan")
                sp_str = f"{speedup:>7.2f}x" if method != "reference" else "      —"
                print(f"{T:>6} {R:>4} {method:>14} {ms:>10.3f} {sp_str}")

            if include_faiss and _HAS_FAISS and numpy_bridge_available():
                cfg = RetrievalConfig(top_k=top_k, max_seq_len=T, method="faiss_flat")
                retriever = RoutingRetriever(cfg).to(device)
                for faiss_m in ("faiss_flat", "faiss_hnsw"):
                    try:
                        ms = time_retriever_method(
                            retriever, q, k, top_k, faiss_m, warmup, runs
                        )
                        speedup = ref_ms / ms if ref_ms else float("nan")
                        print(f"{T:>6} {R:>4} {faiss_m:>14} {ms:>10.3f} {speedup:>7.2f}x")
                    except Exception as exc:
                        print(f"{T:>6} {R:>4} {faiss_m:>14} {'FAIL':>10}  ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark causal top-K retrieval kernels")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=[256, 512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--include-faiss", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Correctness check vs reference:")
    errors = run_correctness(device, args.seq_lens, args.dims, args.top_k, args.batch_size)
    if errors:
        print("\nCORRECTNESS FAILURES:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)
    print("All correctness checks passed.\n")

    if args.correctness_only or args.skip_benchmark:
        return

    run_benchmark(
        device,
        args.seq_lens,
        args.dims,
        args.top_k,
        args.batch_size,
        args.warmup,
        args.runs,
        args.include_faiss,
    )


if __name__ == "__main__":
    main()
