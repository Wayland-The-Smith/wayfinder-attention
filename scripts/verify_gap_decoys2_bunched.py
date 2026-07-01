#!/usr/bin/env python3
"""Verify addr_val + 2 decoys bunched @ T=512 (gap_decoys2 final-ckpt experiment)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.synthetic_protocol import trace_addr_val
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task


def main() -> None:
    print("=== verify_gap_decoys2_bunched ===")
    num_decoys = 2
    payload = generate_synthetic_task(
        "addr_val",
        random.Random(0),
        {
            "num_distractors": num_decoys,
            "synthetic_decoy_addrs": num_decoys,
            "answer_digit_width": 1,
        },
    )
    assert payload.task_type == "addr_val"
    assert len(payload.expected_answer) == 1
    assert len(payload.needle_segments) == 1 + num_decoys
    probe = " ".join(payload.needle_segments)
    addr = int(payload.metadata["addr"])
    assert trace_addr_val(probe, addr) == payload.expected_answer
    print(f"  payload OK: {1 + num_decoys} segments, answer={payload.expected_answer!r}")

    cfg = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[512],
        task_types=["addr_val"],
        needle_depths=[0.25, 0.75],
        suffix_placement="at_end",
        scatter_multi_needles=False,
        num_distractors=num_decoys,
        synthetic_decoy_addrs=num_decoys,
        answer_digit_width=1,
        include_answer_in_suffix=True,
        train_label_mode="answer_only",
    ).apply_synthetic_profile()

    gen = LongContextSampleGenerator(cfg)
    ok = 0
    for seed in range(16):
        sample = gen.generate_one(
            context_length=512,
            needle_depth=0.5,
            task_type="addr_val",
            haystack_mode="synthetic_noise",
            seed=seed,
        )
        assert sample.task_type == "addr_val"
        assert len(sample.input_ids) == 512
        assert not cfg.scatter_multi_needles
        ok += 1
    print(f"  assembled @ T=512: OK ({ok} samples, bunched=True, decoys={num_decoys})")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
