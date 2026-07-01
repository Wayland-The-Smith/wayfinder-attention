#!/usr/bin/env python3
"""Verify dense-gap head-to-head experiment datasets (payload + assembly)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import trace_task_answer
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

CONFIG_DIR = ROOT / "configs" / "dense_gap_headtohead"

EXPERIMENTS: dict[str, Path] = {
    "conflict_scatter_t1024_rows5": CONFIG_DIR / "conflict_scatter_t1024_rows5.yaml",
    "conflict_first_scatter_t2048_rows4": CONFIG_DIR / "conflict_first_scatter_t2048_rows4.yaml",
    "conflict_middle_bunched_t512_rows6": CONFIG_DIR / "conflict_middle_bunched_t512_rows6.yaml",
}


def _bench_from_yaml(path: Path) -> tuple[LongContextBenchmarkConfig, int]:
    arena = load_routing_arena_config(path)
    exp = build_arena_experiment_config(arena, dry_run=False)
    train_t = int(arena["train_context_length"])
    bench = _resolve_synthetic_bench_cfg(exp, train_t)
    return bench, train_t


def _haystack_text(gen: LongContextSampleGenerator, sample) -> str:
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    suffix = sample.question
    if gen.config.include_answer_in_suffix:
        suffix = (
            f"{gen.config.question_prefix}{sample.question}"
            f"{gen.config.answer_prefix}{sample.expected_answer}"
        )
    ss = int(sample.meta_dict["suffix_start"])
    return text[:ss] + text[ss + len(suffix) :]


def verify_experiment(name: str, path: Path) -> None:
    bench, train_t = _bench_from_yaml(path)
    task = bench.task_types[0]
    assert bench.seed != bench.holdout_seed

    kwargs: dict = {
        "synthetic_conflict_rows": bench.synthetic_conflict_rows,
        "answer_digit_width": bench.answer_digit_width,
    }
    payload = generate_synthetic_task(task, random.Random(0), kwargs)
    probe = " ".join(payload.needle_segments)
    assert trace_task_answer(probe, payload) == payload.expected_answer

    gen = LongContextSampleGenerator(bench)
    haystack_mode = "synthetic_noise"
    for seed in range(8):
        sample = gen.generate_one(
            context_length=train_t,
            needle_depth=0.5,
            task_type=task,
            haystack_mode=haystack_mode,
            seed=seed,
        )
        assert sample.context_length == train_t
        assert sample.task_type == task
        haystack = _haystack_text(gen, sample)
        traced = trace_task_answer(haystack, sample)
        assert traced == sample.expected_answer, f"{name} trace mismatch seed={seed}"

    scatter = bench.scatter_multi_needles
    print(
        f"  {name}: OK  task={task}  T={train_t}  "
        f"rows={bench.synthetic_conflict_rows}  scatter={scatter}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    print("=== verify_dense_gap_headtohead ===")
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment {name!r}")
        verify_experiment(name, EXPERIMENTS[name])
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
