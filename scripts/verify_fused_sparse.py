#!/usr/bin/env python3
"""Sanity tests for Layer-3 fused sparse meat attention."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from routing_attention.kernels.causal_topk import causal_topk
from routing_attention.kernels.fused_sparse import (
    SparseMeatConfig,
    fused_sparse_attention,
    sparse_meat_attention,
    sparse_meat_attention_reference,
)


def _rand_inputs(
    B: int,
    H: int,
    T: int,
    D: int,
    R: int,
    K: int,
    device: torch.device,
):
    q_search = torch.randn(B, T, R, device=device)
    k_search = torch.randn(B, T, R, device=device)
    q = torch.randn(B, H, T, D, device=device)
    k = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, D, device=device)
    scale = D ** -0.5
    return q_search, k_search, q, k, v, scale


def check_meat_correctness(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    method: str,
    atol: float = 1e-4,
) -> tuple[bool, float]:
    ref = sparse_meat_attention_reference(q, k, v, indices, scale)
    got = sparse_meat_attention(q, k, v, indices, scale, method=method)
    ok = torch.allclose(ref, got, atol=atol, rtol=1e-4)
    max_diff = (ref - got).abs().max().item()
    return ok, max_diff


def bench_meat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float,
    method: str,
    warmup: int,
    runs: int,
) -> float:
    for _ in range(warmup):
        sparse_meat_attention(q, k, v, indices, scale, method=method)
    if q.is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        sparse_meat_attention(q, k, v, indices, scale, method=method)
    if q.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 784, 2048, 8192])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dims", type=int, nargs="+", default=[64])
    parser.add_argument("--search-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--bench", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    meat_methods = ["reference", "batched", "triton_head", "auto"]
    all_ok = True

    print("\n=== Correctness (meat only) ===")
    for T in args.seq_lens:
        for D in args.dims:
            q_search, k_search, q, k, v, scale = _rand_inputs(
                args.batch, args.heads, T, D, 32, args.top_k, device
            )
            indices = causal_topk(q_search, k_search, args.top_k, method="fused_causal")
            for method in meat_methods:
                if method == "batched" and device.type != "cuda":
                    continue
                if method in ("triton_head", "auto") and device.type != "cuda":
                    continue
                ok, max_diff = check_meat_correctness(q, k, v, indices, scale, method)
                status = "OK" if ok else "FAIL"
                print(f"T={T:>5} D={D} method={method:>12} {status} max_diff={max_diff:.2e}")
                all_ok = all_ok and ok

    print("\n=== Correctness (full fused_sparse_attention) ===")
    for T in [128, 784, 2048] if 2048 in args.seq_lens else [128, 784]:
        for R in args.search_dims:
            D = args.dims[0]
            q_search, k_search, q, k, v, scale = _rand_inputs(
                args.batch, args.heads, T, D, R, args.top_k, device
            )
            ref_idx = causal_topk(q_search, k_search, args.top_k, method="fused_causal")
            ref_out = sparse_meat_attention_reference(q, k, v, ref_idx, scale)
            got_out, got_idx = fused_sparse_attention(
                q_search,
                k_search,
                q,
                k,
                v,
                args.top_k,
                scale,
                retrieval_method="fused_causal",
                meat_method="auto",
            )
            out_ok = torch.allclose(ref_out, got_out, atol=1e-4, rtol=1e-4)
            idx_ok = got_idx.shape == ref_idx.shape
            print(
                f"T={T} R={R} fused_sparse out={'OK' if out_ok else 'FAIL'} "
                f"idx_shape={'OK' if idx_ok else 'FAIL'}"
            )
            all_ok = all_ok and out_ok and idx_ok

    if args.bench and device.type == "cuda":
        print("\n=== Benchmark (meat only, ms) ===")
        print(f"{'T':>6} {'D':>4} {'method':>12} {'ms':>10}")
        print("-" * 36)
        for T in args.seq_lens:
            for D in args.dims:
                q_search, k_search, q, k, v, scale = _rand_inputs(
                    args.batch, args.heads, T, D, 32, args.top_k, device
                )
                indices = causal_topk(q_search, k_search, args.top_k, method="fused_causal")
                for method in meat_methods:
                    ms = bench_meat(q, k, v, indices, scale, method, args.warmup, args.runs)
                    print(f"{T:>6} {D:>4} {method:>12} {ms:>10.3f}")

        print("\n=== Benchmark (full pipeline, ms) ===")
        T = max(args.seq_lens)
        D = args.dims[0]
        R = args.search_dims[0]
        q_search, k_search, q, k, v, scale = _rand_inputs(
            args.batch, args.heads, T, D, R, args.top_k, device
        )
        for meat_method in ["batched", "triton_head", "auto"]:
            for _ in range(args.warmup):
                fused_sparse_attention(
                    q_search, k_search, q, k, v, args.top_k, scale,
                    retrieval_method="fused_causal", meat_method=meat_method,
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.runs):
                fused_sparse_attention(
                    q_search, k_search, q, k, v, args.top_k, scale,
                    retrieval_method="fused_causal", meat_method=meat_method,
                )
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / args.runs * 1000.0
            print(f"T={T} R={R} meat={meat_method} full_pipeline={ms:.3f} ms")

    if not all_ok:
        print("\nFAILED sanity checks")
        sys.exit(1)
    print("\nAll sanity checks passed")


if __name__ == "__main__":
    main()
