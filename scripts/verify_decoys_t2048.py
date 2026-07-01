#!/usr/bin/env python3
"""Smoke-test addr_val decoy scatter @ T=2048 config."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    load_routing_arena_config,
    _resolve_synthetic_bench_cfg,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import trace_addr_val

CFG = ROOT / "configs" / "routing_gap_decoys_scatter_t2048.yaml"


def main() -> None:
    arena = load_routing_arena_config(CFG)
    exp_cfg = build_arena_experiment_config(arena, dry_run=False)
    cfg = _resolve_synthetic_bench_cfg(exp_cfg, 2048)
    gen = LongContextSampleGenerator(cfg)
    for i in range(10):
        s = gen.generate_one(
            context_length=2048,
            task_type="addr_val",
            needle_depth=0.5,
            seed=42 + i,
        )
        assert s.context_length == 2048, s.context_length
        assert s.metadata.get("num_distractors") == 32, s.metadata
        assert s.task_type == "addr_val", s.task_type
    print(
        f"OK: 10 samples @ T=2048  decoys={cfg.num_distractors}  "
        f"scatter={cfg.scatter_multi_needles}  answer_width={cfg.answer_digit_width}"
    )


if __name__ == "__main__":
    main()
