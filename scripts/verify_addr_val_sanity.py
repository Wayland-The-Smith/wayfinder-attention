#!/usr/bin/env python3
"""Smoke-test addr_val sanity config: 1-digit answer, non-scattered, 0 decoys."""

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
    cfg = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[512],
        task_types=["addr_val"],
        needle_depths=[0.25, 0.75],
        suffix_placement="at_end",
        scatter_multi_needles=False,
        num_distractors=0,
        synthetic_decoy_addrs=0,
        answer_digit_width=1,
    ).apply_synthetic_profile()
    payload = generate_synthetic_task(
        "addr_val",
        random.Random(0),
        {
            "num_distractors": 0,
            "synthetic_decoy_addrs": 0,
            "answer_digit_width": 1,
        },
    )
    assert payload.task_type == "addr_val"
    assert len(payload.expected_answer) == 1, payload.expected_answer
    assert payload.expected_answer.isdigit()
    assert len(payload.needle_segments) == 1
    probe = " ".join(payload.needle_segments)
    assert trace_addr_val(probe, int(payload.metadata["addr"])) == payload.expected_answer
    print(f"payload OK: answer={payload.expected_answer!r} segments={len(payload.needle_segments)}")

    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=512,
        needle_depth=0.5,
        task_type="addr_val",
        haystack_mode="synthetic_noise",
        seed=42,
    )
    assert sample.task_type == "addr_val"
    assert len(sample.expected_answer) == 1
    addr = int(sample.metadata["addr"])
    assert sample.metadata.get("value") == sample.expected_answer
    assert len(sample.input_ids) <= 512
    print(f"sample OK: T={len(sample.input_ids)} answer={sample.expected_answer!r} addr={addr}")
    print("addr_val sanity verify OK")


if __name__ == "__main__":
    main()
