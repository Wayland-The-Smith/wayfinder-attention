#!/usr/bin/env python3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from routing_attention.kernels.causal_topk import (
    _causal_topk_heap_during_gemm,
    _causal_topk_row_blocked,
)

T, R, K = 16384, 32, 32
q = torch.randn(1, T, R, device="cuda")
k = torch.randn(1, T, R, device="cuda")

cases = [
    ("row_4096", _causal_topk_row_blocked, {}),
    ("row_8192", _causal_topk_row_blocked, {"block_m": 8192}),
    ("row_16384", _causal_topk_row_blocked, {"block_m": 16384}),
    ("heap_4096", _causal_topk_heap_during_gemm, {}),
]

from routing_attention.kernels.causal_topk import _causal_topk_fused

cases.append(("fused_default", _causal_topk_fused, {}))

for name, fn, kwargs in cases:
    for _ in range(3):
        fn(q, k, K, **kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        fn(q, k, K, **kwargs)
    torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter() - t0) / 20 * 1000:.2f} ms")
