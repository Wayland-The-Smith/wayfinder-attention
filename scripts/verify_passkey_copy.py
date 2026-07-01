#!/usr/bin/env python3
"""Verify passkey_copy payload + assembled samples for the T=4096 experiment."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.holdout import get_holdout_grid
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    PASSKEY_TAG,
    trace_passkey_copy,
    trace_task_answer,
    verify_task_traceable,
)
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

TRAIN_T = 4096
WIDTH = 5


def main() -> None:
    print("=== verify_passkey_copy ===")

    payload = generate_synthetic_task(
        "passkey_copy",
        random.Random(0),
        {"answer_digit_width": WIDTH},
    )
    assert payload.task_type == "passkey_copy"
    assert len(payload.expected_answer) == WIDTH
    probe = " ".join(payload.needle_segments)
    assert trace_passkey_copy(probe) == payload.expected_answer
    assert verify_task_traceable(probe, payload)
    print(f"  payload OK: answer={payload.expected_answer!r} ({WIDTH} digits)")

    bench = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[TRAIN_T],
        task_types=["passkey_copy"],
        needle_depths=[0.25, 0.75],
        suffix_placement="at_end",
        scatter_multi_needles=True,
        answer_digit_width=WIDTH,
        include_answer_in_suffix=True,
        train_label_mode="answer_only",
        seed=42,
        holdout_seed=1_000_042,
    ).apply_synthetic_profile()
    assert bench.seed != bench.holdout_seed
    print(f"  train_seed={bench.seed}  holdout_seed={bench.holdout_seed} (disjoint)")

    gen = LongContextSampleGenerator(bench)
    for seed in range(12):
        sample = gen.generate_one(
            context_length=TRAIN_T,
            needle_depth=0.5,
            task_type="passkey_copy",
            haystack_mode="synthetic_noise",
            seed=seed,
        )
        assert sample.context_length == TRAIN_T
        assert len(sample.expected_answer) == WIDTH
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        ss = int(sample.meta_dict["suffix_start"])
        suffix = sample.question
        if bench.include_answer_in_suffix:
            suffix = (
                f"{bench.question_prefix}{sample.question}"
                f"{bench.answer_prefix}{sample.expected_answer}"
            )
        haystack = text[:ss] + text[ss + len(suffix) :]
        assert f"{PASSKEY_TAG} " in haystack
        assert trace_task_answer(haystack, sample) == sample.expected_answer

    holdout = get_holdout_grid(bench.holdout_config())
    holdout_t = [s for s in holdout if s.context_length == TRAIN_T]
    assert len(holdout_t) > 0
    print(f"  assembled @ T={TRAIN_T}: OK (12 samples, scatter=True)")
    print(f"  holdout grid @ T={TRAIN_T}: {len(holdout_t)} samples (not used in training stream)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
