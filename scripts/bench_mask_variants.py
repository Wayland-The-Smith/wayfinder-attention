#!/usr/bin/env python3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

T, block_m = 16384, 4096
q = torch.randn(1, T, 32, device="cuda", dtype=torch.float32)
k = torch.randn(1, T, 32, device="cuda", dtype=torch.float32)
K = 32

i0, i1 = 0, block_m
bm, i1k = i1 - i0, i1

for _ in range(3):
    torch.bmm(q[:, i0:i1], k[:, :i1].transpose(1, 2))
torch.cuda.synchronize()

# baseline: compare mask + topk
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    scores = torch.bmm(q[:, i0:i1], k[:, :i1].transpose(1, 2))
    row_ids = torch.arange(i0, i1, device="cuda").view(1, bm, 1)
    col_ids = torch.arange(0, i1, device="cuda").view(1, 1, i1)
    scores.masked_fill_(col_ids > row_ids, float("-inf"))
    torch.topk(scores, k=K, dim=-1)
torch.cuda.synchronize()
print(f"compare+mask+topk: {(time.perf_counter()-t0)/50*1000:.2f} ms")

# tril mask
tri = torch.tril(torch.ones(bm, i1k, device="cuda", dtype=torch.bool))
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    scores = torch.bmm(q[:, i0:i1], k[:, :i1].transpose(1, 2))
    scores.masked_fill_(~tri.unsqueeze(0), float("-inf"))
    torch.topk(scores, k=K, dim=-1)
torch.cuda.synchronize()
print(f"tril+topk:         {(time.perf_counter()-t0)/50*1000:.2f} ms")

# topk only (pre-masked scores cached)
scores = torch.bmm(q[:, i0:i1], k[:, :i1].transpose(1, 2))
row_ids = torch.arange(i0, i1, device="cuda").view(1, bm, 1)
col_ids = torch.arange(0, i1, device="cuda").view(1, 1, i1)
scores.masked_fill_(col_ids > row_ids, float("-inf"))
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    torch.topk(scores, k=K, dim=-1, sorted=True)
torch.cuda.synchronize()
print(f"topk sorted:       {(time.perf_counter()-t0)/50*1000:.2f} ms")

t0 = time.perf_counter()
for _ in range(50):
    torch.topk(scores, k=K, dim=-1, sorted=False)
torch.cuda.synchronize()
print(f"topk unsorted:     {(time.perf_counter()-t0)/50*1000:.2f} ms")
