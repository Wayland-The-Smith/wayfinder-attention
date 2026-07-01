#!/usr/bin/env python3
"""Compare linear attention kernel backends in full 6-layer transformer training step."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer, init_experiment_runtime, load_experiment_config
from routing_attention.models import fast_attention as fa


def _step(model, x, labels, opt, V: int) -> None:
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=x)["logits"]
        loss = F.cross_entropy(
            out[:, :-1, :].reshape(-1, V),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    loss.backward()
    opt.step()


def _bench_kernel(name: str, kernel_fn, T: int = 8192, warmup: int = 5, runs: int = 10) -> float:
    orig = fa.fla_causal_linear_attention
    fa.fla_causal_linear_attention = kernel_fn
    try:
        config = load_experiment_config(7)
        device = init_experiment_runtime(config)
        config["model"]["max_seq_len"] = T
        model = build_transformer(config, attention_type="linear").to(device)
        V = model.lm_head.out_features
        x = torch.randint(1, 256, (1, T), device=device)
        labels = x.clone()
        labels[:, : T // 2] = -100
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        for _ in range(warmup):
            _step(model, x, labels, opt, V)
            torch.cuda.synchronize()
        times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _step(model, x, labels, opt, V)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        ms = 1000.0 * sum(times) / len(times)
        print(f"  {name:36s} {ms:7.1f} ms/step")
        return ms
    finally:
        fa.fla_causal_linear_attention = orig


def _chunk(q, k, v, scale):
    from fla.ops.linear_attn import chunk_linear_attn

    q_feat = (F.elu(q * scale) + 1.0).transpose(1, 2).contiguous()
    k_feat = (F.elu(k) + 1.0).transpose(1, 2).contiguous()
    v_bthd = v.transpose(1, 2).contiguous()
    out, _ = chunk_linear_attn(q_feat, k_feat, v_bthd, scale=1.0, head_first=False, normalize=True)
    return out.transpose(1, 2)


def _fused_chunk(q, k, v, scale):
    from fla.ops.linear_attn import fused_chunk_linear_attn

    q_feat = (F.elu(q * scale) + 1.0).transpose(1, 2).contiguous()
    k_feat = (F.elu(k) + 1.0).transpose(1, 2).contiguous()
    v_bthd = v.transpose(1, 2).contiguous()
    out, _ = fused_chunk_linear_attn(q_feat, k_feat, v_bthd, scale=1.0, normalize=True)
    return out.transpose(1, 2)


def _chunk_bf16(q, k, v, scale):
    from fla.ops.linear_attn import chunk_linear_attn

    qf = (F.elu(q.float() * scale) + 1.0).transpose(1, 2).contiguous().to(torch.bfloat16)
    kf = (F.elu(k.float()) + 1.0).transpose(1, 2).contiguous().to(torch.bfloat16)
    vf = v.transpose(1, 2).contiguous().to(torch.bfloat16)
    out, _ = chunk_linear_attn(qf, kf, vf, scale=1.0, head_first=False, normalize=True)
    return out.transpose(1, 2).to(q.dtype)


def _fused_chunk_bf16(q, k, v, scale):
    from fla.ops.linear_attn import fused_chunk_linear_attn

    qf = (F.elu(q.float() * scale) + 1.0).transpose(1, 2).contiguous().to(torch.bfloat16)
    kf = (F.elu(k.float()) + 1.0).transpose(1, 2).contiguous().to(torch.bfloat16)
    vf = v.transpose(1, 2).contiguous().to(torch.bfloat16)
    out, _ = fused_chunk_linear_attn(qf, kf, vf, scale=1.0, normalize=True)
    return out.transpose(1, 2).to(q.dtype)


def main():
    print("Full model linear kernel comparison (T=8192, bf16 AMP)\n")
    _bench_kernel("chunk_fp32", _chunk)
    _bench_kernel("fused_chunk_fp32", _fused_chunk)
    _bench_kernel("chunk_bf16", _chunk_bf16)
    _bench_kernel("fused_chunk_bf16", _fused_chunk_bf16)
    compiled_chunk = torch.compile(_chunk, fullgraph=False)
    _bench_kernel("torch.compile(chunk_fp32)", compiled_chunk)


if __name__ == "__main__":
    main()
