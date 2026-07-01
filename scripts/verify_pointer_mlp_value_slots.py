#!/usr/bin/env python3
"""Verify value-slot labels and pointer_mlp head wiring for slot_pointer."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.slot_pointer import (
    SlotQuad,
    slot_pointer_value_slot_labels,
    trace_value_index,
    verify_slot_pointer_tokens,
)


def test_value_slot_labels() -> None:
    bench = LongContextBenchmarkConfig(num_slot_quads=50).apply_slot_pointer_profile()
    gen = LongContextSampleGenerator(bench)
    rng = random.Random(12345)
    for seed in [rng.randint(0, 2**31 - 1) for _ in range(20)]:
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type="slot_pointer",
            seed=seed,
        )
        meta = sample.meta_dict
        candidates = [int(i) for i in meta["value_candidate_indices"]]
        target_slot = int(meta["pointer_target_slot"])
        target_idx = int(meta["pointer_target_index"])
        assert len(candidates) == 50
        assert 0 <= target_slot < 50
        assert candidates[target_slot] == target_idx
        traced = trace_value_index(sample.input_ids.tolist(), question_index=2047)
        assert traced == target_idx
        verify_slot_pointer_tokens(
            sample.input_ids.tolist(),
            question_index=2047,
            expected_value_index=target_idx,
        )
        recomputed_candidates, recomputed_slot = slot_pointer_value_slot_labels(
            tuple(
                SlotQuad(
                    addr_token=int(q["addr_token"]),
                    value_token=int(q["value_token"]),
                    start_index=int(q["start_index"]),
                )
                for q in meta["metadata"]["quads"]
            ),
            target_idx,
        )
        assert recomputed_candidates == candidates
        assert recomputed_slot == target_slot
    print("  value-slot labels: OK (20 random samples)")


def test_pointer_mlp_forward_and_eval() -> None:
    arena = load_routing_arena_config(ROOT / "configs/routing_slot_pointer_t2048_50q.yaml")
    cfg = build_arena_experiment_config(arena, dry_run=False)
    assert cfg["model"]["output_head"] == "pointer_mlp"
    assert cfg["model"]["pointer_target_mode"] == "value_slots"

    bench = LongContextBenchmarkConfig(num_slot_quads=50).apply_slot_pointer_profile()
    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(context_length=2048, needle_depth=0.5, task_type="slot_pointer", seed=7)
    meta = sample.meta_dict

    model = build_transformer(cfg, attention_type="dense_flash")
    model.eval()
    input_ids = sample.input_ids.unsqueeze(0)
    q_idx = torch.tensor([int(meta["question_index"])], dtype=torch.long)
    slot_target = torch.tensor([int(meta["pointer_target_slot"])], dtype=torch.long)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            question_index=q_idx,
            pointer_target_slot=slot_target,
        )
    assert out["pointer_logits"].shape == (1, 50)
    assert out["loss"].dim() == 0

    evaluator = LongContextEvaluator(bench)
    candidates = [int(i) for i in meta["value_candidate_indices"]]
    target_slot = int(meta["pointer_target_slot"])
    perfect = torch.full((1, 50), -100.0)
    perfect[0, target_slot] = 100.0
    record = evaluator.score_pointer_sample(perfect, meta)
    assert record.correct, f"perfect logits failed: pred={record.predicted} expected={record.expected}"
    assert int(record.predicted) == candidates[target_slot]
    print("  pointer_mlp forward + perfect-eval: OK")


def main() -> None:
    print("=== pointer_mlp value_slots verification ===")
    test_value_slot_labels()
    test_pointer_mlp_forward_and_eval()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
