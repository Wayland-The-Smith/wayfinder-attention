#!/usr/bin/env python3
"""
Benchmark routing vector search at various sequence lengths.

Usage:
  python scripts/benchmark_retrieval.py
  python scripts/benchmark_retrieval.py --seq-lens 512 2048 8192 32768
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from routing_attention.evaluation.benchmarking import benchmark_routing_retrieval
from routing_attention.utils.config import load_config, merge_configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--routing-dim", type=int, default=32)
    args = parser.parse_args()

    base = load_config(ROOT / "configs" / "base.yaml")
    retrieval_cfg = base.get("retrieval", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Retrieval config: {retrieval_cfg}\n")

    print(f"{'seq_len':>10} {'method':>14} {'ms':>10}")
    print("-" * 38)
    for seq_len in args.seq_lens:
        out = benchmark_routing_retrieval(
            seq_len=seq_len,
            batch_size=1,
            routing_dim=args.routing_dim,
            top_k=args.top_k,
            device=device,
            retrieval_cfg=retrieval_cfg,
        )
        method = out["resolved_method"]
        ms = out["retrieval_ms_per_call"].get(method, float("nan"))
        print(f"{seq_len:>10} {method:>14} {ms:>10.3f}")
        for m, t in out["retrieval_ms_per_call"].items():
            if m != method:
                print(f"{'':>10} {m:>14} {t:>10.3f}  (alt)")


if __name__ == "__main__":
    main()
