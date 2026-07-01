#!/usr/bin/env python3
"""Smoke-test generation for all benchmark variants."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.slot_pointer import (
    SlotQuad,
    slot_pointer_value_slot_labels,
    trace_value_index,
    verify_slot_pointer_tokens,
)


def _load_variant_bench(name: str, variant: dict, suite: dict) -> LongContextBenchmarkConfig:
    slot_defaults = dict(suite.get("slot_pointer_defaults", {}))
    bench = {**slot_defaults, **variant.get("long_context_benchmark", {})}
    cfg = LongContextBenchmarkConfig.from_dict(bench)
    tasks = list(cfg.task_types or [])
    if tasks == ["slot_pointer"]:
        cfg = cfg.apply_slot_pointer_profile()
    else:
        cfg = cfg.apply_synthetic_profile()
    return cfg


def main() -> None:
    suite_path = ROOT / "configs" / "benchmark_variants_suite.yaml"
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))["benchmark_variants_suite"]
    variants = suite["variants"]
    rng = random.Random(42)

    for name, variant in variants.items():
        cfg = _load_variant_bench(name, variant, suite)
        gen = LongContextSampleGenerator(cfg)
        task = cfg.task_types[0]
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type=task,
            seed=rng.randint(0, 2**31 - 1),
        )
        meta = sample.meta_dict
        if task == "slot_pointer":
            target = int(meta["pointer_target_index"])
            slot = int(meta["pointer_target_slot"])
            cands = [int(i) for i in meta["value_candidate_indices"]]
            assert cands[slot] == target
            traced = trace_value_index(sample.input_ids.tolist(), question_index=2047)
            assert traced == target
            verify_slot_pointer_tokens(
                sample.input_ids.tolist(),
                question_index=2047,
                expected_value_index=target,
            )
            _, slot2 = slot_pointer_value_slot_labels(
                tuple(
                    SlotQuad(
                        addr_token=int(q["addr_token"]),
                        value_token=int(q["value_token"]),
                        start_index=int(q["start_index"]),
                    )
                    for q in meta["metadata"]["quads"]
                ),
                target,
            )
            assert slot2 == slot
            print(f"  {name}: slot_pointer OK (quads={len(cands)} placement={meta.get('slot_quad_placement')})")
        else:
            assert meta.get("include_answer_in_suffix") is False
            assert meta.get("label_mode") == "query_only_answer"
            assert meta["expected_answer"]
            print(f"  {name}: pointer_unique OK (query-only, answer not in suffix)")

    print("ALL VARIANT GENERATION CHECKS PASSED")


if __name__ == "__main__":
    print("=== verify benchmark variants ===")
    main()
