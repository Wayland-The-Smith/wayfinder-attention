#!/usr/bin/env python3
"""Verify slot_pointer dataset generation, tracing, and procedural diversity."""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.slot_pointer import (
    HAY_TOKENS,
    SEMANTIC_TOKENS,
    TOK_COMMA,
    TOK_SEMICOLON,
    generate_slot_pointer_tokens,
    trace_addr_index,
    trace_value_index,
    verify_slot_pointer_tokens,
)


def test_low_level_generator() -> None:
    rng = random.Random(0)
    for num_quads in (1, 10, 50, 100):
        for _ in range(20):
            slot = generate_slot_pointer_tokens(rng, context_length=2048, num_quads=num_quads)
            assert len(slot.tokens) == 2048
            assert slot.question_index == 2047
            assert slot.tokens[2047] == slot.query_addr_token
            assert slot.tokens[2047] in SEMANTIC_TOKENS
            traced = verify_slot_pointer_tokens(
                slot.tokens,
                question_index=slot.question_index,
                expected_value_index=slot.target_value_index,
            )
            assert traced == slot.target_value_index
            assert slot.tokens[traced] == slot.target_value_token
            assert slot.tokens[traced - 2] == slot.query_addr_token
            assert slot.tokens[traced - 1] == TOK_SEMICOLON
            assert slot.tokens[traced + 1] == TOK_COMMA
            addr_hit = trace_addr_index(slot.tokens, slot.question_index)
            assert addr_hit == traced - 2
    print("  low-level generator: OK")


def test_integration_generator() -> None:
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        task_types=["slot_pointer"],
        num_slot_quads=50,
    ).apply_slot_pointer_profile()
    gen = LongContextSampleGenerator(cfg)
    seen_layouts: set[tuple[int, ...]] = set()
    for i in range(200):
        sample = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type="slot_pointer",
            seed=10_000 + i,
        )
        assert sample.task_type == "slot_pointer"
        assert sample.input_ids.shape[0] == 2048
        assert int(sample.meta_dict["question_index"]) == 2047
        target = int(sample.meta_dict["pointer_target_index"])
        tokens = sample.input_ids.tolist()
        traced = verify_slot_pointer_tokens(
            tokens,
            question_index=2047,
            expected_value_index=target,
        )
        assert traced == target
        assert tokens[2047] == int(sample.meta_dict["query_addr_token"])
        assert tokens[target] == int(sample.meta_dict["pointer_target_token"])
        seen_layouts.add(tuple(tokens))
    assert len(seen_layouts) == 200, "expected unique layouts across 200 seeds"
    print("  integration generator: OK (200 unique layouts / 200 seeds)")


def test_infinite_diversity() -> None:
    """Different seeds should almost always yield different (query, target) pairs."""
    cfg = LongContextBenchmarkConfig(num_slot_quads=50).apply_slot_pointer_profile()
    gen = LongContextSampleGenerator(cfg)
    keys: set[tuple[int, int, int]] = set()
    for seed in range(5000):
        s = gen.generate_one(
            context_length=2048,
            needle_depth=0.5,
            task_type="slot_pointer",
            seed=seed,
        )
        keys.add(
            (
                int(s.meta_dict["query_addr_token"]),
                int(s.meta_dict["pointer_target_index"]),
                int(s.meta_dict["pointer_target_token"]),
            )
        )
    assert len(keys) > 4500, f"expected high diversity, got {len(keys)} unique triples"
    print(f"  diversity: OK ({len(keys)} unique query/index/value triples in 5000 seeds)")


def test_vocabulary_constraints() -> None:
    rng = random.Random(123)
    slot = generate_slot_pointer_tokens(rng, context_length=512, num_quads=20)
    content = slot.tokens[: slot.question_index]
    hay_count = sum(1 for t in content if t in HAY_TOKENS)
    quad_tokens = 20 * 4
    assert hay_count == len(content) - quad_tokens
    delim_counts = Counter(content)
    assert delim_counts[TOK_SEMICOLON] == 20
    assert delim_counts[TOK_COMMA] == 20
    print("  vocabulary / hay fill: OK")


def test_holdout_grid() -> None:
    from pathlib import Path

    from routing_attention.benchmarks.long_context.holdout import resolve_holdout_splits
    from routing_attention.benchmarks.long_context.routing_arena import (
        _resolve_synthetic_bench_cfg,
        build_arena_experiment_config,
        load_routing_arena_config,
    )

    arena = load_routing_arena_config(ROOT / "configs/routing_slot_pointer_t2048_50q.yaml")
    cfg = build_arena_experiment_config(arena, dry_run=False)
    bench = _resolve_synthetic_bench_cfg(cfg, 2048)
    _, full, _meta = resolve_holdout_splits(cfg, bench, 2048)
    assert len(full) == 300
    for sample in full[:10]:
        assert sample.task_type == "slot_pointer"
        verify_slot_pointer_tokens(
            sample.input_ids.tolist(),
            question_index=2047,
            expected_value_index=int(sample.meta_dict["pointer_target_index"]),
        )
    print("  holdout grid: OK (300 samples, spot-checked 10)")


def test_pointer_head() -> None:
    import torch

    from experiments.common import build_transformer
    from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
    from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
    from routing_attention.benchmarks.long_context.routing_arena import (
        build_arena_experiment_config,
        load_routing_arena_config,
    )

    arena = load_routing_arena_config(ROOT / "configs/routing_slot_pointer_t2048_50q.yaml")
    cfg = build_arena_experiment_config(arena, dry_run=False)
    assert cfg["model"]["output_head"] == "pointer_mlp"
    assert cfg["model"]["pointer_target_mode"] == "value_slots"
    assert cfg["model"]["vocab_size"] == 129

    bench = LongContextBenchmarkConfig(num_slot_quads=50).apply_slot_pointer_profile()
    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(context_length=512, needle_depth=0.5, task_type="slot_pointer", seed=42)
    meta = sample.meta_dict
    input_ids = sample.input_ids.unsqueeze(0)
    q_idx = torch.tensor([int(meta["question_index"])], dtype=torch.long)
    target = torch.tensor([int(meta["pointer_target_slot"])], dtype=torch.long)

    model = build_transformer(cfg, attention_type="dense_flash")
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            question_index=q_idx,
            pointer_target_slot=target,
        )
    assert out["pointer_logits"].shape == (1, 50)
    assert out["loss"].dim() == 0
    assert 0.0 <= float(out["pointer_accuracy"]) <= 1.0

    evaluator = LongContextEvaluator(bench)
    record = evaluator.score_sample(out["pointer_logits"], meta)
    assert record.task_type == "slot_pointer"
    print("  pointer head forward/eval: OK")


def main() -> None:
    print("=== slot_pointer dataset verification ===")
    test_low_level_generator()
    test_integration_generator()
    test_infinite_diversity()
    test_vocabulary_constraints()
    test_holdout_grid()
    test_pointer_head()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
