#!/usr/bin/env python3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from routing_attention.kernels.causal_topk import (
    _causal_topk_tiled_gemm,
    _finalize_causal_topk,
    _merge_tile_topk,
)

T = 512
q = torch.randn(1, T, 32, device="cuda").float()
k = torch.randn(1, T, 32, device="cuda").float()
k_eff = 32
block_j = 512

out_idx = torch.full((1, T, k_eff), -1, dtype=torch.int32, device="cuda")
out_val = torch.full((1, T, k_eff), float("-inf"), device="cuda")

for _ in range(3):
    torch.matmul(q, k[:, :block_j].transpose(1, 2))
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    torch.matmul(q, k[:, :block_j].transpose(1, 2))
torch.cuda.synchronize()
print(f"matmul only: {(time.perf_counter()-t0)/100*1000:.3f}ms")

scores = torch.matmul(q, k[:, :block_j].transpose(1, 2))
for _ in range(3):
    _merge_tile_topk(out_val, out_idx, scores, 0, k_eff)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    ov, oi = _merge_tile_topk(out_val, out_idx, scores, 0, k_eff)
torch.cuda.synchronize()
print(f"merge only: {(time.perf_counter()-t0)/100*1000:.3f}ms")

for _ in range(3):
    _finalize_causal_topk(ov, oi, k_eff)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _finalize_causal_topk(ov, oi, k_eff)
torch.cuda.synchronize()
print(f"finalize only: {(time.perf_counter()-t0)/100*1000:.3f}ms")

for _ in range(3):
    _causal_topk_tiled_gemm(q, k, k_eff)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _causal_topk_tiled_gemm(q, k, k_eff)
torch.cuda.synchronize()
print(f"full tiled: {(time.perf_counter()-t0)/100*1000:.3f}ms")
