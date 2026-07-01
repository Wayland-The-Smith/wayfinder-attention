#!/usr/bin/env python3
"""Verify dense-gap feasibility experiment datasets (payload + assembly)."""

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
from routing_attention.benchmarks.long_context.tasks import trace_nl_task_answer
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

CONFIG_DIR = ROOT / "configs" / "dense_gap_feasibility"

EXPERIMENTS: dict[str, Path] = {
    "addr_val_conflict_bunched_t512": CONFIG_DIR / "addr_val_conflict_bunched_t512.yaml",
    "addr_val_conflict_first_bunched_t512": CONFIG_DIR / "addr_val_conflict_first_bunched_t512.yaml",
    "addr_val_conflict_scatter_t1024": CONFIG_DIR / "addr_val_conflict_scatter_t1024.yaml",
    "addr_val_conflict_first_scatter_t1024": CONFIG_DIR / "addr_val_conflict_first_scatter_t1024.yaml",
    "passkey_distractor_bunched_t512": CONFIG_DIR / "passkey_distractor_bunched_t512.yaml",
    "pointer_unique_copy_bunched_t512": CONFIG_DIR / "pointer_unique_copy_bunched_t512.yaml",
    "nl_exact_retrieval_t512": CONFIG_DIR / "nl_exact_retrieval_t512.yaml",
    "nl_distractor_t1024": CONFIG_DIR / "nl_distractor_t1024.yaml",
}


def _bench_from_yaml(path: Path) -> tuple[LongContextBenchmarkConfig, int, str]:
    arena = load_routing_arena_config(path)
    exp = build_arena_experiment_config(arena, dry_run=False)
    train_t = int(arena["train_context_length"])
    family = str(exp.get("long_context_benchmark", {}).get("benchmark_family", "synthetic"))
    if family == "synthetic":
        bench = _resolve_synthetic_bench_cfg(exp, train_t)
    else:
        bench = LongContextBenchmarkConfig.from_dict(exp["long_context_benchmark"]).normalized()
    return bench, train_t, family


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
    bench, train_t, family = _bench_from_yaml(path)
    task = bench.task_types[0]
    assert bench.seed != bench.holdout_seed

    if family == "synthetic":
        kwargs: dict = {
            "synthetic_conflict_rows": bench.synthetic_conflict_rows,
            "answer_digit_width": bench.answer_digit_width,
            "num_distractors": bench.num_distractors,
            "synthetic_decoy_keys": bench.synthetic_decoy_keys,
        }
        payload = generate_synthetic_task(task, random.Random(0), kwargs)
        probe = " ".join(payload.needle_segments)
        assert trace_task_answer(probe, payload) == payload.expected_answer

    haystack_mode = "random_tokens" if family == "nl" else "synthetic_noise"
    gen = LongContextSampleGenerator(bench)
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
        if family == "synthetic":
            traced = trace_task_answer(haystack, sample)
        else:
            traced = trace_nl_task_answer(haystack, sample)
        assert traced == sample.expected_answer, f"{name} trace mismatch seed={seed}"

    scatter = bench.scatter_multi_needles
    print(
        f"  {name}: OK  task={task}  T={train_t}  "
        f"scatter={scatter}  family={family}"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all")
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    print("=== verify_dense_gap_feasibility ===")
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"Unknown experiment {name!r}")
        verify_experiment(name, EXPERIMENTS[name])
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
