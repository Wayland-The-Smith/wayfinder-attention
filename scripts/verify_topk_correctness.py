#!/usr/bin/env python3
"""Verify fused matches reference on causal-valid top-K scores."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from routing_attention.kernels.causal_topk import (
    _causal_topk_row_blocked,
    causal_topk_reference,
)


def valid_score_multiset(q, k, idx, t, T):
    row_idx = idx[0, t].long().clamp(0, T - 1)
    scores = (q[0, t].float() * k[0, row_idx].float()).sum(-1)
    causal = row_idx <= t
    scores = scores.masked_fill(~causal, float("-inf"))
    finite = scores[scores.isfinite() & (scores > -1e30)]
    return finite.sort().values


for T in [512, 4096, 16384]:
    q = torch.randn(1, T, 32, device="cuda")
    k = torch.randn(1, T, 32, device="cuda")
    ref = causal_topk_reference(q, k, 32)
    got = _causal_topk_row_blocked(q, k, 32)
    bad = 0
    for t in range(T):
        rs = valid_score_multiset(q, k, ref, t, T)
        gs = valid_score_multiset(q, k, got, t, T)
        if not torch.allclose(rs, gs, atol=1e-4, rtol=0):
            bad += 1
    print(f"T={T}: mismatched rows {bad}/{T}")
