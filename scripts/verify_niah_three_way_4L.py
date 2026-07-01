#!/usr/bin/env python3
"""Sanity-check 4L three-way NIAH diagnostic config and sample generation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.holdout import resolve_holdout_splits
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import trace_task_answer
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

CONFIG = ROOT / "configs" / "niah_three_way_4L" / "niah_pointer_unique_t2048.yaml"


def main() -> None:
    arena = load_routing_arena_config(CONFIG)
    exp = build_arena_experiment_config(arena, dry_run=False, n_layers=4)
    train_t = int(arena["train_context_length"])
    bench = _resolve_synthetic_bench_cfg(exp, train_t)

    assert int(arena["n_layers"]) == 4
    assert bench.task_types == ["pointer_unique"]
    assert bench.scatter_multi_needles is False
    assert bench.synthetic_decoy_keys == 0
    assert bench.train_label_mode == "answer_only"
    assert bench.include_answer_in_suffix is True
    assert exp["model"]["n_layers"] == 4

    cal = exp.get("dense_calibration", {})
    assert cal.get("restore_best_checkpoint") is True
    assert cal.get("eval_use_full_holdout") is True

    payload = generate_synthetic_task(
        "pointer_unique",
        random.Random(42),
        {"synthetic_decoy_keys": 0, "answer_digit_width": 1},
    )
    assert trace_task_answer(" ".join(payload.needle_segments), payload) == payload.expected_answer

    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(
        context_length=train_t,
        needle_depth=0.5,
        task_type="pointer_unique",
        haystack_mode="synthetic_noise",
        seed=99,
    )
    assert sample.task_type == "pointer_unique"

    _, holdout_full, meta = resolve_holdout_splits(exp, bench, train_t)
    assert meta["holdout_total_target"] == 300
    assert len(holdout_full) == 300

    steps = int(exp["transformer"]["sparse_finetune_steps"])
    assert steps == 20000

    print("=== verify_niah_three_way_4L ===")
    print(f"  T={train_t}  n_layers=4  steps={steps}")
    print(f"  task=pointer_unique  scatter=false  decoys=0")
    print(f"  holdout={len(holdout_full)}  restore_best=true")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
