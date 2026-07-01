#!/usr/bin/env python3
"""Sanity checks for mqar_addr_val dataset generation and assembled samples."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from routing_attention.benchmarks.long_context.config import (
    LongContextBenchmarkConfig,
    apply_synthetic_family_profile,
)
from routing_attention.benchmarks.long_context.generator import (
    LongContextSampleGenerator,
    _filler_text,
    _haystack_has_collision,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    ADDR_TAG,
    QUERY_TAG,
    VAL_TAG,
    trace_addr_val,
    trace_task_answer,
    verify_task_traceable,
)
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

CFG_PATH = ROOT / "configs" / "routing_mqar_addr_val_calibration.yaml"
KV_COUNTS = (2, 8, 16, 24)
SAMPLES_PER_N = 12
CONTEXT_LENGTH = 512


def _decode_text(tokenizer, ids) -> str:
    return tokenizer.decode(ids.tolist() if hasattr(ids, "tolist") else list(ids))


def _gold_row_pattern(addr: int, value: str) -> str:
    return f"{ADDR_TAG} {addr} {VAL_TAG} {value}."


def _verify_payload(rng: random.Random, num_kv: int) -> None:
    payload = generate_synthetic_task(
        "mqar_addr_val",
        rng,
        {
            "num_kv_pairs": num_kv,
            "num_queries": 1,
            "answer_digit_width": 1,
        },
    )
    assert payload.task_type == "mqar_addr_val"
    meta = payload.metadata
    assert meta["num_kv_pairs"] == num_kv
    assert meta["answer_digit_width"] == 1
    assert meta["protocol"] == "mqar_addr_val_v1"
    assert len(meta["bindings"]) == num_kv
    assert len(payload.needle_segments) == num_kv
    addrs = {b["addr"] for b in meta["bindings"]}
    assert len(addrs) == num_kv

    probe = " ".join(payload.needle_segments)
    assert verify_task_traceable(probe, payload)
    assert trace_task_answer(probe, payload) == payload.expected_answer

    target_addr = int(meta["query_addrs"][-1])
    assert payload.question.startswith(f"{QUERY_TAG} ")
    assert str(target_addr) in payload.question
    assert trace_addr_val(probe, target_addr) == payload.expected_answer


def _verify_assembled_sample(
    gen: LongContextSampleGenerator,
    *,
    num_kv: int,
    seed: int,
) -> None:
    sample = gen.generate_one(
        context_length=CONTEXT_LENGTH,
        needle_depth=0.5,
        task_type="mqar_addr_val",
        haystack_mode="synthetic_noise",
        seed=seed,
    )
    meta = sample.meta_dict
    task_meta = meta["metadata"]
    bindings = task_meta["bindings"]
    assert sample.task_type == "mqar_addr_val"
    assert len(sample.input_ids) == CONTEXT_LENGTH
    assert meta["label_mode"] == "query_only_answer"
    assert meta["include_answer_in_suffix"] is False
    assert meta["scatter_needles"] is False

    q_idx = int(meta["question_index"])
    assert q_idx == CONTEXT_LENGTH - 1
    value_tok = int(meta["query_only_answer_token"])
    assert int(sample.labels[q_idx].item()) == value_tok

    text = _decode_text(gen.tokenizer, sample.input_ids)
    assert len(text) == CONTEXT_LENGTH

    from routing_attention.benchmarks.long_context.tasks import TaskPayload

    trace_payload = TaskPayload(
        needle_segments=meta["needle_segments"],
        question=meta["question"],
        expected_answer=meta["expected_answer"],
        task_type="mqar_addr_val",
        metadata=task_meta,
    )
    traced = trace_task_answer(text, trace_payload)
    assert traced == meta["expected_answer"], f"trace {traced!r} != {meta['expected_answer']!r}"
    assert verify_task_traceable(text, trace_payload)

    target_addr = int(task_meta["query_addrs"][-1])
    gold_pattern = _gold_row_pattern(target_addr, meta["expected_answer"])
    gold_idx = text.find(gold_pattern)
    assert 0 <= gold_idx < q_idx, "gold row must appear before query (causal reachability)"

    for binding in bindings:
        addr = int(binding["addr"])
        value = str(binding["value"])
        row = _gold_row_pattern(addr, value)
        assert row in text, f"missing binding row for addr={addr}"
        assert trace_addr_val(text, addr) == value

    # Hay filler must not contain semantic collision tokens (haystack only — not query suffix)
    suffix_start = int(meta["suffix_start"])
    question = meta["question"]
    hay_only = text[:suffix_start] + text[suffix_start + len(question) :]
    filler = _filler_text(hay_only, meta["needle_segments"])
    assert not _haystack_has_collision(filler, trace_payload)

    # Supervised token must decode to expected answer digit
    pred_char = gen.tokenizer.decode([value_tok])
    assert pred_char == meta["expected_answer"]

    # Question suffix sits at sequence end
    assert text.endswith(meta["question"])


def _load_calibration_bench(num_kv: int) -> LongContextBenchmarkConfig:
    raw = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))["routing_arena"]
    bench = dict(raw["long_context_benchmark"])
    bench["num_kv_pairs"] = num_kv
    cfg = LongContextBenchmarkConfig.from_dict(bench)
    return apply_synthetic_family_profile(cfg)


def main() -> int:
    rng = random.Random(0)
    print("=== mqar_addr_val payload checks ===")
    for num_kv in KV_COUNTS:
        for _ in range(4):
            _verify_payload(rng, num_kv)
        print(f"  N={num_kv}: payload OK (4 samples)")

    print("=== mqar_addr_val assembled sample checks ===")
    for num_kv in KV_COUNTS:
        bench = _load_calibration_bench(num_kv)
        assert bench.task_types == ["mqar_addr_val"]
        assert bench.scatter_multi_needles is False
        assert bench.train_label_mode == "query_only_answer"
        assert bench.answer_digit_width == 1
        gen = LongContextSampleGenerator(bench)
        for i in range(SAMPLES_PER_N):
            _verify_assembled_sample(gen, num_kv=num_kv, seed=10_000 + num_kv * 100 + i)
        print(f"  N={num_kv}: assembled OK ({SAMPLES_PER_N} samples @ T={CONTEXT_LENGTH})")

    print("=== config profile ===")
    bench0 = _load_calibration_bench(2)
    assert bench0.include_answer_in_suffix is False
    assert bench0.num_queries == 1
    print(f"  profile OK: variant={bench0.benchmark_variant!r}")

    print("ALL mqar_addr_val sanity checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
