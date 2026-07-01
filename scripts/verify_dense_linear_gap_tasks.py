#!/usr/bin/env python3
"""Verify dense-vs-linear gap task generators (payload + assembled samples)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    PASSKEY_TAG,
    trace_addr_val_first,
    trace_addr_val_last,
    trace_addr_val_middle,
    trace_passkey_copy,
    trace_passkey_distractor,
    trace_pointer_unique,
    trace_task_answer,
    verify_task_traceable,
)
from routing_attention.benchmarks.long_context.tasks import (
    generate_task,
    trace_distractor,
    trace_exact_retrieval,
    trace_nl_task_answer,
)
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

DEPTHS = (0.10, 0.50, 0.90)
ASSEMBLE_T = 2048
SAMPLES_PER_TASK = 12


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


def _bench_cfg(
    task_type: str,
    *,
    family: str = "synthetic",
    scatter: bool = True,
    num_distractors: int = 4,
    num_conflict_rows: int = 3,
    answer_digit_width: int = 1,
) -> LongContextBenchmarkConfig:
    if family == "nl":
        return LongContextBenchmarkConfig(
            benchmark_family="nl",
            context_lengths=[ASSEMBLE_T],
            task_types=[task_type],
            needle_depths=list(DEPTHS),
            suffix_placement="at_end",
            scatter_multi_needles=scatter,
            num_distractors=num_distractors,
            include_answer_in_suffix=True,
            train_label_mode="answer_only",
            haystack_modes=["random_tokens"],
        ).normalized()

    cfg = LongContextBenchmarkConfig(
        benchmark_family="synthetic",
        context_lengths=[ASSEMBLE_T],
        task_types=[task_type],
        needle_depths=list(DEPTHS),
        suffix_placement="at_end",
        scatter_multi_needles=scatter,
        num_distractors=num_distractors,
        synthetic_decoy_keys=num_distractors,
        synthetic_decoy_addrs=num_distractors,
        synthetic_conflict_rows=num_conflict_rows,
        answer_digit_width=answer_digit_width,
        include_answer_in_suffix=True,
        train_label_mode="answer_only",
    )
    return cfg.apply_synthetic_profile()


def _verify_payload_synthetic(task_type: str, kwargs: dict) -> None:
    rng = random.Random(17)
    for i in range(8):
        payload = generate_synthetic_task(task_type, rng, kwargs)
        assert payload.task_type == task_type, (task_type, payload.task_type)
        probe = " ".join(payload.needle_segments)
        assert verify_task_traceable(probe, payload), f"{task_type} trace failed @ {i}"
        assert trace_task_answer(probe, payload) == payload.expected_answer
        assert payload.expected_answer, f"{task_type} empty answer"
        assert payload.question, f"{task_type} empty question"
    print(f"  {task_type} payloads: OK")


def _verify_assembled(task_type: str, kwargs: dict, *, family: str = "synthetic") -> None:
    scatter = kwargs.get("scatter", True)
    num_distractors = int(kwargs.get("num_distractors", 4))
    num_conflict_rows = int(kwargs.get("synthetic_conflict_rows", 3))
    answer_digit_width = int(kwargs.get("answer_digit_width", 1))
    bench = _bench_cfg(
        task_type,
        family=family,
        scatter=scatter,
        num_distractors=num_distractors,
        num_conflict_rows=num_conflict_rows,
        answer_digit_width=answer_digit_width,
    )
    gen = LongContextSampleGenerator(bench)
    haystack_mode = "random_tokens" if family == "nl" else "synthetic_noise"

    for i, depth in enumerate(DEPTHS * 4):
        sample = gen.generate_one(
            context_length=ASSEMBLE_T,
            needle_depth=depth,
            task_type=task_type,
            haystack_mode=haystack_mode,
            seed=5000 + i,
        )
        assert sample.context_length == ASSEMBLE_T
        assert sample.task_type == task_type
        assert sample.answer_end > sample.answer_start
        haystack = _haystack_text(gen, sample)
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        assert sample.question in text

        if family == "synthetic":
            assert trace_task_answer(haystack, sample) == sample.expected_answer, (
                f"{task_type} assembled trace mismatch depth={depth}"
            )
            if task_type == "pointer_unique":
                key = str(sample.metadata["key"])
                assert haystack.count(f"{key} ") == 1
            if task_type == "pointer_unique_copy":
                key = str(sample.metadata["key"])
                assert haystack.count(f"{key} ") == 1
                assert len(sample.expected_answer) >= 2
            if task_type in ("passkey_copy", "passkey_distractor"):
                assert haystack.count(f"{PASSKEY_TAG} ") >= 1
            if task_type.startswith("addr_val_conflict"):
                addr = int(sample.metadata["addr"])
                n = int(sample.metadata["num_conflict_rows"])
                assert haystack.count(f"ADDR {addr} VAL") == n
        else:
            traced = trace_nl_task_answer(haystack, sample)
            assert traced == sample.expected_answer, (
                f"{task_type} NL assembled trace mismatch depth={depth}"
            )

        nd = float(sample.meta_dict.get("needle_depth_final", sample.meta_dict["needle_depth"]))
        assert 0.0 <= nd <= 1.0

    print(
        f"  {task_type} assembled @ T={ASSEMBLE_T}: OK "
        f"({SAMPLES_PER_TASK} samples, scatter={scatter})"
    )


def _verify_nl_payload(task_type: str, kwargs: dict) -> None:
    rng = random.Random(23)
    for i in range(8):
        payload = generate_task(task_type, rng, kwargs)
        assert payload.task_type == task_type
        probe = " ".join(payload.needle_segments)
        if task_type == "exact_retrieval":
            assert trace_exact_retrieval(probe) == payload.expected_answer
            assert len(payload.expected_answer) == 5
        elif task_type == "distractor":
            target = str(payload.metadata["target"])
            assert trace_distractor(probe, target) == target
            assert len(payload.needle_segments) == 1 + int(payload.metadata["num_distractors"])
        assert trace_nl_task_answer(probe, payload) == payload.expected_answer
    print(f"  {task_type} payloads: OK")


def _verify_conflict_policies() -> None:
    probe = " ".join(
        ["ADDR 42 VAL 1.", "ADDR 42 VAL 2.", "ADDR 42 VAL 3.", "ADDR 42 VAL 4.", "ADDR 42 VAL 5."]
    )
    assert trace_addr_val_last(probe, 42) == "5"
    assert trace_addr_val_first(probe, 42) == "1"
    assert trace_addr_val_middle(probe, 42) == "3"
    rng = random.Random(99)
    for task, trace_fn in (
        ("addr_val_conflict", trace_addr_val_last),
        ("addr_val_conflict_first", trace_addr_val_first),
        ("addr_val_conflict_middle", trace_addr_val_middle),
    ):
        payload = generate_synthetic_task(
            task,
            rng,
            {"synthetic_conflict_rows": 5, "answer_digit_width": 1},
        )
        p = " ".join(payload.needle_segments)
        addr = int(payload.metadata["addr"])
        assert trace_fn(p, addr) == payload.expected_answer, task
    print("  conflict policy ordering: OK")


def _verify_passkey_distractor_uniqueness() -> None:
    rng = random.Random(31)
    payload = generate_synthetic_task(
        "passkey_distractor",
        rng,
        {"num_distractors": 6, "answer_digit_width": 5},
    )
    probe = " ".join(payload.needle_segments)
    target = str(payload.metadata["target"])
    assert trace_passkey_distractor(probe, target) == target
    assert probe.count(f"{PASSKEY_TAG} {target}.") == 1
    print("  passkey_distractor uniqueness: OK")


def main() -> None:
    print("=== verify_dense_linear_gap_tasks ===\n")

    print("[synthetic selective-copy]")
    _verify_payload_synthetic(
        "passkey_copy",
        {"answer_digit_width": 5},
    )
    _verify_assembled("passkey_copy", {"answer_digit_width": 5, "scatter": True})

    _verify_payload_synthetic(
        "passkey_distractor",
        {"num_distractors": 6, "answer_digit_width": 5},
    )
    _verify_assembled(
        "passkey_distractor",
        {"num_distractors": 6, "answer_digit_width": 5, "scatter": True},
    )
    _verify_passkey_distractor_uniqueness()

    _verify_payload_synthetic(
        "pointer_unique_copy",
        {"num_distractors": 4, "synthetic_decoy_keys": 4, "answer_digit_width": 4},
    )
    _verify_assembled(
        "pointer_unique_copy",
        {
            "num_distractors": 4,
            "synthetic_decoy_keys": 4,
            "answer_digit_width": 4,
            "scatter": True,
        },
    )

    print("\n[synthetic addr_val conflict variants]")
    _verify_conflict_policies()
    for task, policy_check in (
        ("addr_val_conflict", trace_addr_val_last),
        ("addr_val_conflict_first", trace_addr_val_first),
        ("addr_val_conflict_middle", trace_addr_val_middle),
    ):
        _verify_payload_synthetic(
            task,
            {"synthetic_conflict_rows": 4, "answer_digit_width": 1},
        )
        rng = random.Random(7)
        payload = generate_synthetic_task(
            task,
            rng,
            {"synthetic_conflict_rows": 4, "answer_digit_width": 1},
        )
        probe = " ".join(payload.needle_segments)
        addr = int(payload.metadata["addr"])
        assert policy_check(probe, addr) == payload.expected_answer
        _verify_assembled(
            task,
            {"synthetic_conflict_rows": 4, "answer_digit_width": 1, "scatter": True},
        )

    print("\n[NL passkey / distractor]")
    _verify_nl_payload("exact_retrieval", {})
    _verify_assembled("exact_retrieval", {"num_distractors": 0, "scatter": True}, family="nl")
    _verify_nl_payload("distractor", {"num_distractors": 8})
    _verify_assembled(
        "distractor",
        {"num_distractors": 8, "scatter": True},
        family="nl",
    )

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
