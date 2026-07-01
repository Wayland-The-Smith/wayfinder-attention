#!/usr/bin/env python3
"""Flash (SDPA) vs naive dense vs fused sparse — scaling benchmark on GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from experiments.common import build_transformer, load_experiment_config, init_experiment_runtime
from routing_attention.evaluation.benchmarking import benchmark_attention
from routing_attention.utils.config import merge_configs, load_config


def sdpa_backend_status() -> dict:
    status = {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    if hasattr(torch.backends, "cuda"):
        for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
            fn = getattr(torch.backends.cuda, name, None)
            if fn is not None:
                status[name.replace("_enabled", "")] = fn()
    return status


def verify_sdpa_matches_naive(T: int, device: torch.device) -> bool:
    from routing_attention.models.attention import DenseAttention, DenseSDPAAttention

    d_model, n_heads = 256, 4
    x = torch.randn(1, T, d_model, device=device)
    naive = DenseAttention(d_model, n_heads).to(device)
    flash = DenseSDPAAttention(d_model, n_heads).to(device)
    flash.load_state_dict(naive.state_dict())
    naive.eval()
    flash.eval()
    with torch.no_grad():
        out_naive = naive(x)
        out_flash = flash(x)
    return torch.allclose(out_naive, out_flash, atol=1e-4, rtol=1e-3)


def load_variant_model(variant: str, max_seq: int, device: torch.device, retrieval: dict):
    base = load_config(ROOT / "configs" / "base.yaml")
    exp4 = load_config(ROOT / "configs" / "experiment_4.yaml")
    exp6 = load_config(ROOT / "configs" / "experiment_6.yaml")
    cfg = merge_configs(merge_configs(base, exp4), exp6)
    cfg["model"]["max_seq_len"] = max_seq
    cfg["retrieval"] = {**cfg.get("retrieval", {}), **retrieval, "max_seq_len": max_seq}

    from experiments.experiment_4 import ATTENTION_VARIANTS, _apply_variant_config, _load_state_dict_tolerant
    from routing_attention.utils.experiment import resolve_checkpoint_path

    var_cfg = _apply_variant_config(cfg, variant)
    attn_type = ATTENTION_VARIANTS.get(variant, variant)
    if attn_type == "routing":
        from experiments.common import load_router_from_reuse
        router, _ = load_router_from_reuse(var_cfg, device)
        model = build_transformer(var_cfg, attention_type="routing", router=router).to(device)
    else:
        model = build_transformer(var_cfg, attention_type=attn_type).to(device)

    ckpt_map = {
        "routing_asymmetric": "Experiment_4/run_023/checkpoints/routing_asymmetric_final.pt",
        "dense": "Experiment_4/run_026/checkpoints/dense_final.pt",
        "dense_flash": "Experiment_4/run_026/checkpoints/dense_final.pt",
    }
    ckpt = resolve_checkpoint_path(ckpt_map.get(variant, ""))
    if ckpt and Path(ckpt).exists():
        _load_state_dict_tolerant(model, Path(ckpt), device)

    if max_seq > 784:
        from experiments.experiment_4 import _expand_position_embeddings
        _expand_position_embeddings(model, max_seq)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[784, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--variants", type=str, nargs="+", default=["dense", "dense_flash", "routing_asymmetric"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    print("=== SDPA backend status ===")
    for k, v in sdpa_backend_status().items():
        print(f"  {k}: {v}")

    print("\n=== SDPA vs naive correctness (T=128) ===")
    ok = verify_sdpa_matches_naive(128, torch.device("cuda"))
    print(f"  match: {'OK' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)

    retrieval_sparse = {
        "method": "fused_causal",
        "apply_to_key_vector": True,
        "use_fused_sparse": True,
        "dtype": "float16",
        "use_gpu": True,
    }
    retrieval_none = {}

    print(f"\n=== Full-model forward latency (ms) — {args.runs} runs ===")
    header = f"{'T':>6} " + " ".join(f"{v:>18}" for v in args.variants)
    print(header)
    print("-" * len(header))

    results: dict[int, dict[str, float | None]] = {}

    for T in args.seq_lens:
        results[T] = {}
        row = f"{T:>6}"
        for variant in args.variants:
            ret = retrieval_sparse if variant == "routing_asymmetric" else retrieval_none
            try:
                model = load_variant_model(variant, T, torch.device("cuda"), ret)
                bench = benchmark_attention(
                    model,
                    seq_len=T,
                    batch_size=1,
                    device=torch.device("cuda"),
                    num_warmup=args.warmup,
                    num_runs=args.runs,
                    retrieval_cfg=ret,
                )
                ms = bench["latency_ms"]
                results[T][variant] = ms
                row += f" {ms:>18.2f}"
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[T][variant] = None
                    row += f" {'OOM':>18}"
                    torch.cuda.empty_cache()
                else:
                    raise
        print(row)

    print("\n=== Sparse vs Flash speedup (routing_asymmetric / dense_flash) ===")
    for T in args.seq_lens:
        sparse = results[T].get("routing_asymmetric")
        flash = results[T].get("dense_flash")
        naive = results[T].get("dense")
        if sparse is not None and flash is not None:
            print(f"  T={T:>5}: sparse={sparse:.2f} ms  flash={flash:.2f} ms  ratio={flash/sparse:.2f}x (flash/sparse)")
        if naive is not None and flash is not None:
            print(f"           naive={naive:.2f} ms  flash={flash:.2f} ms  flash speedup={naive/flash:.2f}x")


if __name__ == "__main__":
    main()
