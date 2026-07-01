"""Smoke-test massive_addr_val generation and trace at T=2048."""

from __future__ import annotations

import random
import sys

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    trace_task_answer,
    verify_task_traceable,
)
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task


def main() -> int:
    rng = random.Random(0)
    payload = generate_synthetic_task("massive_addr_val", rng, {"num_kv_pairs": 50})
    assert payload.task_type == "massive_addr_val"
    assert payload.metadata["num_kv_pairs"] == 50
    probe = " ".join(payload.needle_segments)
    assert trace_task_answer(probe, payload) == payload.expected_answer

    cfg = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[2048],
        task_types=["massive_addr_val"],
        num_kv_pairs=50,
        num_distractors=0,
    ).apply_synthetic_profile()
    gen = LongContextSampleGenerator(cfg)
    sample = gen.generate_one(
        context_length=2048,
        needle_depth=0.5,
        task_type="massive_addr_val",
        haystack_mode="synthetic_noise",
        seed=42,
    )
    assert sample.task_type == "massive_addr_val"
    assert sample.metadata.get("num_kv_pairs") == 50
    assert len(sample.input_ids) <= 2048
    print(
        f"ok: T=2048 len={len(sample.input_ids)} "
        f"kv_pairs={sample.metadata.get('num_kv_pairs', 0)} "
        f"answer={sample.expected_answer!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
