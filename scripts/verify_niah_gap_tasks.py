#!/usr/bin/env python3
"""Validate pointer_unique NIAH @ T=4096 and scattered multi-query MQAR samples."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from routing_attention.benchmarks.long_context.config import apply_synthetic_family_profile
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import (
    LongContextSampleGenerator,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    load_routing_arena_config,
    _resolve_synthetic_bench_cfg,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import (
    ADDR_TAG,
    QUERY_TAG,
    VAL_TAG,
    trace_addr_val,
    trace_pointer_unique,
    trace_task_answer,
    verify_task_traceable,
)
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task

POINTER_CFG = ROOT / "configs" / "routing_pointer_unique_niah_t4096.yaml"
MQAR_CFG = ROOT / "configs" / "routing_mqar_scatter_multiquery_t2048.yaml"
SAMPLES_PER_CHECK = 16
DEPTHS = (0.10, 0.25, 0.50, 0.75, 0.90)


def _bench_from_arena(path: Path, train_t: int) -> LongContextBenchmarkConfig:
    arena = load_routing_arena_config(path)
    exp = build_arena_experiment_config(arena, dry_run=False)
    return _resolve_synthetic_bench_cfg(exp, train_t)


def _haystack_text(gen: LongContextSampleGenerator, sample) -> str:
    text = gen.tokenizer.decode(sample.input_ids.tolist())
    suffix = sample.question
    if gen.config.include_answer_in_suffix:
        suffix = f"{gen.config.question_prefix}{sample.question}{gen.config.answer_prefix}{sample.expected_answer}"
    ss = int(sample.meta_dict["suffix_start"])
    return text[:ss] + text[ss + len(suffix) :]


def verify_pointer_unique_payload() -> None:
    rng = random.Random(42)
    for depth in DEPTHS:
        for i in range(8):
            payload = generate_synthetic_task(
                "pointer_unique",
                rng,
                {"num_distractors": 4, "synthetic_decoy_keys": 4},
            )
            probe = " ".join(payload.needle_segments)
            key = str(payload.metadata["key"])
            assert verify_task_traceable(probe, payload)
            assert trace_pointer_unique(probe, key) == payload.expected_answer
            assert len(payload.needle_segments) == 5
    print("  pointer_unique payloads: OK")


def verify_pointer_unique_assembled(train_t: int = 4096) -> None:
    bench = _bench_from_arena(POINTER_CFG, train_t)
    assert bench.scatter_multi_needles is True
    assert bench.task_types == ["pointer_unique"]
    gen = LongContextSampleGenerator(bench)
    depths_seen: set[float] = set()
    for i, depth in enumerate(DEPTHS * 3):
        sample = gen.generate_one(
            context_length=train_t,
            needle_depth=depth,
            task_type="pointer_unique",
            haystack_mode="synthetic_noise",
            seed=1000 + i,
        )
        assert sample.context_length == train_t
        assert sample.task_type == "pointer_unique"
        key = str(sample.metadata["key"])
        haystack = _haystack_text(gen, sample)
        assert haystack.count(f"{key} ") == 1, f"key {key!r} must appear once in haystack"
        assert sample.metadata.get("num_distractors", 0) >= 1
        assert sample.meta_dict.get("scatter_needles") is True
        assert sample.answer_end > sample.answer_start
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        assert f"Q {key}" in text
        assert f"Q {key} A {sample.expected_answer}" in text
        assert int(sample.meta_dict.get("query_to_needle_distance", 0)) >= 0
        nd = float(sample.meta_dict.get("needle_depth_final", sample.meta_dict["needle_depth"]))
        assert 0.0 <= nd <= 1.0
        segments = sample.meta_dict.get("needle_segments") or sample.metadata.get("needle_segments", [])
        filler = haystack
        for seg in segments:
            filler = filler.replace(seg, " ")
        for forbidden in sample.metadata.get("collision_checks", []):
            if isinstance(forbidden, str) and len(forbidden) > 2:
                assert forbidden not in filler or forbidden in " ".join(segments)
        depths_seen.add(round(float(sample.meta_dict["needle_depth"]), 2))
    print(
        f"  pointer_unique assembled @ T={train_t}: OK "
        f"({SAMPLES_PER_CHECK} samples, scatter={bench.scatter_multi_needles}, "
        f"decoys={bench.num_distractors})"
    )


def verify_mqar_payload() -> None:
    rng = random.Random(99)
    for n, q in ((16, 8), (12, 4)):
        payload = generate_synthetic_task(
            "mqar_addr_val",
            rng,
            {
                "num_kv_pairs": n,
                "num_queries": q,
                "answer_digit_width": 1,
            },
        )
        meta = payload.metadata
        assert meta["num_kv_pairs"] == n
        assert meta["num_queries"] == q
        assert len(meta["query_addrs"]) == q
        probe = " ".join(payload.needle_segments)
        target = int(meta["query_addrs"][-1])
        assert trace_addr_val(probe, target) == payload.expected_answer
        assert verify_task_traceable(probe, payload)
    print("  mqar_addr_val payloads: OK")


def verify_mqar_assembled(train_t: int = 2048) -> None:
    bench = _bench_from_arena(MQAR_CFG, train_t)
    assert bench.scatter_multi_needles is True
    assert bench.num_kv_pairs == 16
    assert bench.num_queries == 8
    assert bench.train_label_mode == "query_only_answer"
    gen = LongContextSampleGenerator(bench)
    for i, depth in enumerate(DEPTHS * 3):
        sample = gen.generate_one(
            context_length=train_t,
            needle_depth=depth,
            task_type="mqar_addr_val",
            haystack_mode="synthetic_noise",
            seed=2000 + i,
        )
        assert sample.context_length == train_t
        assert sample.task_type == "mqar_addr_val"
        meta = sample.metadata
        assert meta["num_kv_pairs"] == 16
        assert meta["num_queries"] == 8
        assert len(meta["query_addrs"]) == 8
        assert sample.meta_dict.get("scatter_needles") is True
        assert sample.meta_dict["label_mode"] == "query_only_answer"
        assert sample.meta_dict.get("query_only_answer_token") is not None
        text = gen.tokenizer.decode(sample.input_ids.tolist())
        for addr in meta["query_addrs"]:
            assert f"{QUERY_TAG} {addr}" in sample.question
        target_addr = int(meta["query_addrs"][-1])
        probe = _haystack_text(gen, sample)
        segments = sample.meta_dict.get("needle_segments") or sample.metadata.get("needle_segments", [])
        for seg in segments:
            assert seg in probe or seg.replace(".", "") in probe
        assert trace_addr_val(probe, target_addr) == sample.expected_answer
        q_idx = int(sample.meta_dict["question_index"])
        assert q_idx == train_t - 1
    print(
        f"  mqar scatter multi-query @ T={train_t}: OK "
        f"(N={bench.num_kv_pairs}, Q={bench.num_queries}, scatter=True)"
    )


def main() -> None:
    print("=== verify_niah_gap_tasks ===")
    print("\n[pointer_unique NIAH @ T=4096]")
    verify_pointer_unique_payload()
    verify_pointer_unique_assembled(4096)
    print("\n[mqar scatter N=16 Q=8 @ T=2048]")
    verify_mqar_payload()
    verify_mqar_assembled(2048)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
