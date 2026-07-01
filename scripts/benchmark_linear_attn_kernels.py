#!/usr/bin/env python3
"""Benchmark FLA linear attention kernel variants for Experiment 7."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fla.ops.linear_attn import (
    chunk_linear_attn,
    fused_chunk_linear_attn,
    fused_recurrent_linear_attn,
)


def _features(B: int, H: int, T: int, D: int, device: torch.device, dtype: torch.dtype):
    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    scale = D ** -0.5
    q_feat = (F.elu(q * scale) + 1.0).transpose(1, 2).contiguous()
    k_feat = (F.elu(k) + 1.0).transpose(1, 2).contiguous()
    v_bthd = v.transpose(1, 2).contiguous()
    return q_feat, k_feat, v_bthd


def _bench(name: str, fn, warmup: int = 5, runs: int = 15) -> float | None:
    try:
        for _ in range(warmup):
            qf, kf, vf = fn.inputs()
            qf = qf.detach().requires_grad_(True)
            kf = kf.detach().requires_grad_(True)
            vf = vf.detach().requires_grad_(True)
            out = fn.run(qf, kf, vf)
            out.sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            qf, kf, vf = fn.inputs()
            qf = qf.detach().requires_grad_(True)
            kf = kf.detach().requires_grad_(True)
            vf = vf.detach().requires_grad_(True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn.run(qf, kf, vf)
            out.sum().backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        ms = 1000.0 * sum(times) / len(times)
        print(f"  {name:32s} {ms:8.2f} ms")
        return ms
    except Exception as e:
        print(f"  {name:32s} FAILED: {e}")
        return None


class _Kernel:
    def __init__(self, B, H, T, D, device, dtype, runner):
        self.B, self.H, self.T, self.D = B, H, T, D
        self.device, self.dtype = device, dtype
        self.runner = runner

    def inputs(self):
        return _features(self.B, self.H, self.T, self.D, self.device, self.dtype)

    def run(self, qf, kf, vf):
        return self.runner(qf, kf, vf)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__}\n")

    configs = [(1, 4, 8192, 64), (4, 4, 8192, 64)]
    for dtype in (torch.float32, torch.bfloat16):
        print(f"===== dtype={dtype} =====")
        for B, H, T, D in configs:
            print(f"--- B={B} H={H} T={T} D={D} ---")

            def chunk_run(qf, kf, vf):
                out, _ = chunk_linear_attn(qf, kf, vf, scale=1.0, head_first=False, normalize=True)
                return out.transpose(1, 2)

            def fused_chunk_run(qf, kf, vf):
                out, _ = fused_chunk_linear_attn(qf, kf, vf, scale=1.0, normalize=True)
                return out.transpose(1, 2)

            def fused_recurrent_run(qf, kf, vf):
                out, _ = fused_recurrent_linear_attn(qf, kf, vf, scale=1.0, normalize=True)
                return out.transpose(1, 2)

            _bench("chunk_linear_attn", _Kernel(B, H, T, D, device, dtype, chunk_run))
            _bench("fused_chunk_linear_attn", _Kernel(B, H, T, D, device, dtype, fused_chunk_run))
            _bench("fused_recurrent_linear_attn", _Kernel(B, H, T, D, device, dtype, fused_recurrent_run))
            print()


if __name__ == "__main__":
    main()
