#!/usr/bin/env python3
"""Break down row-blocked fused kernel time at T=16384."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

T, R, K, B = 16384, 32, 32, 1
block_m = 4096
q = torch.randn(B, T, R, device="cuda", dtype=torch.float32)
k = torch.randn(B, T, R, device="cuda", dtype=torch.float32)
kt = k.transpose(1, 2).contiguous()
col_idx = torch.arange(T, device="cuda")

# warmup
for _ in range(3):
    torch.bmm(q[:, :block_m], kt)
torch.cuda.synchronize()

# full brute baseline
for _ in range(5):
    s = torch.bmm(q, kt)
    s = s.masked_fill(~torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool)).unsqueeze(0), float("-inf"))
    torch.topk(s, k=K, dim=-1)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    s = torch.bmm(q, kt)
    s = s.masked_fill(~torch.tril(torch.ones(T, T, device="cuda", dtype=torch.bool)).unsqueeze(0), float("-inf"))
    torch.topk(s, k=K, dim=-1)
torch.cuda.synchronize()
print(f"brute (GEMM+mask+topk): {(time.perf_counter()-t0)/20*1000:.2f} ms")

# GEMM only full
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    torch.bmm(q, kt)
torch.cuda.synchronize()
print(f"full GEMM only:          {(time.perf_counter()-t0)/20*1000:.2f} ms")

# row-blocked breakdown
matmul_ms = mask_ms = topk_ms = 0.0
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i0 in range(0, T, block_m):
        i1 = min(i0 + block_m, T)
        scores = torch.bmm(q[:, i0:i1], kt)
    torch.cuda.synchronize()
    matmul_ms += (time.perf_counter() - t0) * 50  # /20

for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i0 in range(0, T, block_m):
        i1 = min(i0 + block_m, T)
        scores = torch.bmm(q[:, i0:i1], kt)
        row_ids = torch.arange(i0, i1, device="cuda").view(1, -1, 1)
        scores.masked_fill_(col_idx.view(1, 1, T) > row_ids, float("-inf"))
    torch.cuda.synchronize()
    mask_ms += (time.perf_counter() - t0) * 50

for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i0 in range(0, T, block_m):
        i1 = min(i0 + block_m, T)
        scores = torch.bmm(q[:, i0:i1], kt)
        row_ids = torch.arange(i0, i1, device="cuda").view(1, -1, 1)
        scores.masked_fill_(col_idx.view(1, 1, T) > row_ids, float("-inf"))
        torch.topk(scores, k=K, dim=-1)
    torch.cuda.synchronize()
    topk_ms += (time.perf_counter() - t0) * 50

print(f"row-block matmul (4x):    {matmul_ms:.2f} ms")
print(f"row-block + mask:        {mask_ms:.2f} ms")
print(f"row-block + mask + topk: {topk_ms:.2f} ms")

# fp16 matmul row-block
qh, kh = q.half(), k.half()
kht = kh.transpose(1, 2).contiguous()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    for i0 in range(0, T, block_m):
        i1 = min(i0 + block_m, T)
        scores = torch.bmm(qh[:, i0:i1], kht).float()
        row_ids = torch.arange(i0, i1, device="cuda").view(1, -1, 1)
        scores.masked_fill_(col_idx.view(1, 1, T) > row_ids, float("-inf"))
        torch.topk(scores, k=K, dim=-1)
torch.cuda.synchronize()
print(f"row-block fp16 matmul:   {(time.perf_counter()-t0)/20*1000:.2f} ms")

# Theoretical min: bytes moved / bandwidth
bytes_qk = B * T * R * 4 * 2  # read Q and K once
bytes_scores_avoided = B * T * T * 4  # brute writes this
print(f"\nQ+K read: {bytes_qk/1e6:.0f} MB | brute score write avoided: {bytes_scores_avoided/1e6:.0f} MB")
