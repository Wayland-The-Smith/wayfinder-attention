#!/usr/bin/env python3
"""Verify hop-first scattered synthetic protocol on 30 samples per task."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import (
    LongContextSampleGenerator,
    _filler_text,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    PROTOCOL_VERSION,
    trace_task_answer,
    verify_task_traceable,
)
from routing_attention.benchmarks.long_context.tasks import TaskPayload
from routing_attention.benchmarks.long_context.tasks_synthetic import SYNTHETIC_ALL_TASK_TYPES

SAMPLES_PER_TASK = 30


def _haystack_text(gen, sample) -> str:
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    suffix = (
        f"{gen.config.question_prefix}{sample.question}"
        f"{gen.config.answer_prefix}{sample.expected_answer}"
    )
    ss = int(sample.meta_dict["suffix_start"])
    return text[:ss] + text[ss + len(suffix) :]


def _needle_centers(haystack: str, segments: list[str]) -> list[float]:
    centers: list[float] = []
    for seg in segments:
        idx = haystack.find(seg)
        if idx >= 0:
            centers.append(idx + len(seg) / 2.0)
    return centers


def _payload(sample) -> TaskPayload:
    return TaskPayload(
        needle_segments=list(sample.meta_dict.get("needle_segments", [])),
        question=sample.question,
        expected_answer=sample.expected_answer,
        task_type=sample.task_type,
        metadata=dict(sample.metadata),
    )


def main() -> None:
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        task_types=list(SYNTHETIC_ALL_TASK_TYPES),
        needle_depths=[0.5],
        num_distractors=6,
        synthetic_hop_count=3,
        synthetic_decoy_keys=6,
        synthetic_decoy_addrs=6,
        scatter_placement_min=0,
        scatter_placement_max=None,
        suffix_placement="at_end",
    ).apply_synthetic_profile()
    gen = LongContextSampleGenerator(cfg)

    failures: list[str] = []
    for task in SYNTHETIC_ALL_TASK_TYPES:
        spans: list[float] = []
        last_seg_count = 0
        for seed in range(SAMPLES_PER_TASK):
            sample = gen.generate_one(
                context_length=2048,
                needle_depth=0.5,
                task_type=task,
                seed=10_000 + seed,
            )
            haystack = _haystack_text(gen, sample)
            segments = sample.meta_dict.get("needle_segments", [])
            last_seg_count = len(segments)

            if int(sample.meta_dict["suffix_start"]) != len(haystack):
                failures.append(f"{task} seed={seed}: query not at_end")
            if sample.metadata.get("protocol_version") != PROTOCOL_VERSION:
                failures.append(f"{task} seed={seed}: wrong protocol_version")
            if not sample.metadata.get("trace_verified"):
                failures.append(f"{task} seed={seed}: trace_verified flag missing")

            payload = _payload(sample)
            if not verify_task_traceable(haystack, payload):
                failures.append(
                    f"{task} seed={seed}: trace failed expected={sample.expected_answer!r} "
                    f"got={trace_task_answer(haystack, payload)!r}"
                )

            filler = _filler_text(haystack, segments)
            if re.search(r"[A-Za-z0-9]", filler):
                failures.append(f"{task} seed={seed}: filler has alphanumeric content")

            for seg in segments:
                if haystack.count(seg) != 1:
                    failures.append(f"{task} seed={seed}: segment count != 1 for {seg!r}")

            centers = _needle_centers(haystack, segments)
            if len(centers) >= 2:
                spans.append(max(centers) - min(centers))

            if task == "ptr_chain" and sample.metadata.get("hop_count") != cfg.synthetic_hop_count:
                failures.append(f"{task} seed={seed}: hop_count mismatch")

        if last_seg_count >= 2 and spans:
            mean_span = sum(spans) / len(spans)
            if mean_span < 200:
                failures.append(
                    f"{task}: needles too clustered (mean span={mean_span:.0f} < 200)"
                )
        print(f"{task}: OK {SAMPLES_PER_TASK} samples mean_needle_span={sum(spans)/len(spans):.0f}" if spans else f"{task}: OK {SAMPLES_PER_TASK} samples")

    if failures:
        print("\nFAILURES:")
        for f in failures[:40]:
            print(f"  - {f}")
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more")
        raise SystemExit(1)

    print(f"\nAll checks passed ({len(SYNTHETIC_ALL_TASK_TYPES)} tasks x {SAMPLES_PER_TASK} samples).")


if __name__ == "__main__":
    main()
