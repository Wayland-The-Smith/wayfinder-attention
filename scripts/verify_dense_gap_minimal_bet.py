#!/usr/bin/env python3
"""Sanity-check minimal-bet dense-vs-linear experiment datasets."""

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

CONFIG_DIR = ROOT / "configs" / "dense_gap_minimal_bet"

EXPERIMENTS: dict[str, Path] = {
    "niah_diagnostic_t2048": CONFIG_DIR / "niah_diagnostic_t2048.yaml",
    "pointer_1decoy_first_wins_t2048": CONFIG_DIR / "pointer_1decoy_first_wins_t2048.yaml",
    "mqar_n4_q4_t2048": CONFIG_DIR / "mqar_n4_q4_t2048.yaml",
}


def _bench_for(name: str, path: Path):
    arena = load_routing_arena_config(path)
    exp = build_arena_experiment_config(arena, dry_run=False)
    train_t = int(arena["train_context_length"])
    return _resolve_synthetic_bench_cfg(exp, train_t), train_t, arena


def verify_niah_diagnostic() -> None:
    bench, train_t, _ = _bench_for("niah_diagnostic_t2048", EXPERIMENTS["niah_diagnostic_t2048"])
    assert bench.task_types == ["pointer_unique"]
    assert bench.scatter_multi_needles is False
    assert bench.synthetic_decoy_keys == 0
    assert bench.train_label_mode == "answer_only"
    assert bench.include_answer_in_suffix is True

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
    assert sample.meta_dict.get("label_mode") == "answer_only"
    print(f"  niah_diagnostic_t2048 @ T={train_t}: OK")


def verify_pointer_1decoy_first_wins() -> None:
    bench, train_t, _ = _bench_for(
        "pointer_1decoy_first_wins_t2048",
        EXPERIMENTS["pointer_1decoy_first_wins_t2048"],
    )
    assert bench.task_types == ["pointer_conflict_first"]
    assert bench.scatter_multi_needles is True
    assert bench.synthetic_conflict_rows == 2
    assert bench.synthetic_decoy_keys == 1

    rng = random.Random(7)
    for _ in range(16):
        payload = generate_synthetic_task(
            "pointer_conflict_first",
            rng,
            {
                "synthetic_conflict_rows": 2,
                "synthetic_decoy_keys": 1,
                "answer_digit_width": 1,
            },
        )
        assert len(payload.needle_segments) == 3
        probe = " ".join(payload.needle_segments)
        assert trace_task_answer(probe, payload) == payload.expected_answer

    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(
        context_length=train_t,
        needle_depth=0.5,
        task_type="pointer_conflict_first",
        haystack_mode="synthetic_noise",
        seed=123,
    )
    assert sample.task_type == "pointer_conflict_first"
    assert len(sample.meta_dict.get("needle_segments", [])) >= 2
    print(f"  pointer_1decoy_first_wins_t2048 @ T={train_t}: OK (3 segments, scatter)")


def verify_mqar_n4_q4() -> None:
    bench, train_t, arena = _bench_for("mqar_n4_q4_t2048", EXPERIMENTS["mqar_n4_q4_t2048"])
    assert bench.task_types == ["mqar_addr_val"]
    assert bench.num_kv_pairs == 4
    assert bench.num_queries == 4
    assert bench.mqar_supervise_all_queries is True
    assert bench.include_answer_in_suffix is True
    assert bench.train_label_mode == "answer_only"

    payload = generate_synthetic_task(
        "mqar_addr_val",
        random.Random(11),
        {
            "num_kv_pairs": 4,
            "num_queries": 4,
            "mqar_supervise_all_queries": True,
            "answer_digit_width": 1,
        },
    )
    assert payload.metadata.get("mqar_supervise_all_queries") is True
    assert len(payload.metadata.get("query_addrs", [])) == 4
    assert " " in payload.expected_answer
    probe = " ".join(payload.needle_segments)
    assert trace_task_answer(probe, payload) == payload.expected_answer

    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(
        context_length=train_t,
        needle_depth=0.5,
        task_type="mqar_addr_val",
        haystack_mode="synthetic_noise",
        seed=55,
    )
    assert sample.task_type == "mqar_addr_val"
    assert " " in sample.expected_answer

    exp = build_arena_experiment_config(
        load_routing_arena_config(EXPERIMENTS["mqar_n4_q4_t2048"]),
        dry_run=False,
    )
    _, holdout_full, meta = resolve_holdout_splits(exp, bench, train_t)
    assert meta["holdout_total_target"] == 300
    print(f"  mqar_n4_q4_t2048 @ T={train_t}: OK (4 answers supervised, holdout={len(holdout_full)})")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment.strip()]
    print("=== verify_dense_gap_minimal_bet ===")
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment {name!r}")
        if name == "niah_diagnostic_t2048":
            verify_niah_diagnostic()
        elif name == "pointer_1decoy_first_wins_t2048":
            verify_pointer_1decoy_first_wins()
        elif name == "mqar_n4_q4_t2048":
            verify_mqar_n4_q4()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
