#!/usr/bin/env python3
"""Fast WSL benchmark: reference / brute_force / fused_causal only (no streaming)."""

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
    has_triton,
)


def time_method(q, k, top_k, method, warmup, runs):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 2048, 8192, 16384])
    parser.add_argument("--dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GPU: {torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu'}")
    print(f"torch={torch.__version__}  triton={has_triton()}  fused_available={causal_topk_available()}")
    print(f"top_k={args.top_k}  runs={args.runs}\n")

    methods = ["brute_force", "fused_causal"]
    # Per-row Triton heap is correctness/debug only at long T (T launches).
    if causal_topk_available() and max(args.seq_lens) <= 4096:
        methods.append("triton")

    print(f"{'T':>6} {'R':>4} {'method':>14} {'ms':>10} {'vs_ref':>8}")
    print("-" * 48)

    for T in args.seq_lens:
        for R in args.dims:
            q = torch.randn(1, T, R, device=device)
            k = torch.randn(1, T, R, device=device)

            if "fused_causal" in methods:
                ref = causal_topk_reference(q, k, args.top_k)
                got = causal_topk(q, k, args.top_k, method="fused_causal")

                def causal_scores(indices, t):
                    row_idx = indices[0, t].long().clamp(0, T - 1)
                    s = (q[0, t].float() * k[0, row_idx].float()).sum(-1)
                    s = s.masked_fill(row_idx > t, float("-inf"))
                    finite = s[s.isfinite() & (s > -1e30)]
                    return finite.sort().values

                sample = [0, min(T - 1, 31), min(T - 1, T // 2), T - 1]
                if not all(
                    torch.allclose(causal_scores(ref, t), causal_scores(got, t), atol=1e-4)
                    for t in sample
                ):
                    print(f"WARN correctness T={T} R={R} fused_causal mismatch")

            ref_ms = time_method(q, k, args.top_k, "brute_force", args.warmup, args.runs)
            print(f"{T:>6} {R:>4} {'reference':>14} {ref_ms:>10.3f} {'—':>8}")

            for method in methods:
                if method == "brute_force":
                    continue
                ms = time_method(q, k, args.top_k, method, args.warmup, args.runs)
                speedup = ref_ms / ms
                print(f"{T:>6} {R:>4} {method:>14} {ms:>10.3f} {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
