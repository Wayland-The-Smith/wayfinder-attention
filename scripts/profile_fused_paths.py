#!/usr/bin/env python3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from routing_attention.kernels.causal_topk import (
    _causal_topk_fused,
    _causal_topk_heap_micro,
    _causal_topk_key_tiled,
    _causal_topk_row_blocked,
    causal_topk_reference,
)

for T in [512, 8192, 16384]:
    q = torch.randn(1, T, 32, device="cuda")
    k = torch.randn(1, T, 32, device="cuda")
    for name, fn in [
        ("brute", lambda: causal_topk_reference(q, k, 32)),
        ("heap_gemm", lambda: _causal_topk_heap_micro(q, k, 32)),
        ("row_blocked", lambda: _causal_topk_row_blocked(q, k, 32)),
        ("key_tiled", lambda: _causal_topk_key_tiled(q, k, 32)),
        ("fused", lambda: _causal_topk_fused(q, k, 32)),
    ]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        print(f"T={T:>5} {name:>12} {(time.perf_counter()-t0)/20*1000:>8.2f} ms")
