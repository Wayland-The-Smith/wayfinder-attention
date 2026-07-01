#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import torch
from routing_attention.kernels.causal_topk import causal_topk, causal_topk_reference

for T in [512, 4096, 16384]:
    q = torch.randn(1, T, 32, device="cuda")
    k = torch.randn(1, T, 32, device="cuda")
    ref = causal_topk_reference(q, k, 32)
    methods = ["brute_force", "fused_causal"]
    from routing_attention.kernels.causal_topk import causal_topk_available
    if causal_topk_available():
        methods.append("triton")
    for method in methods:
        for _ in range(3):
            causal_topk(q, k, 32, method=method)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            got = causal_topk(q, k, 32, method=method)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 10 * 1000
        def causal_scores(idx, t):
            row_idx = idx[0, t].long().clamp(0, T - 1)
            s = (q[0, t].float() * k[0, row_idx].float()).sum(-1)
            s = s.masked_fill(row_idx > t, float("-inf"))
            finite = s[s.isfinite() & (s > -1e30)]
            return finite.sort().values

        ok = all(
            torch.allclose(causal_scores(ref, t), causal_scores(got, t), atol=1e-4)
            for t in range(T)
        )
        print(f"T={T:>5} {method:>14} {ms:>8.2f}ms  ok={ok}")
