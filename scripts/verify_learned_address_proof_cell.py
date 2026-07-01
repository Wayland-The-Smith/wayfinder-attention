#!/usr/bin/env python3
"""Sanity-check learned-address proof cell config and sample generation."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learned_address_proof_common import CONFIG_PATH, PROOF_VARIANTS, ROUTING_TOP_K
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.holdout import resolve_holdout_splits
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.synthetic_protocol import trace_task_answer
from routing_attention.benchmarks.long_context.tasks_synthetic import generate_synthetic_task


def main() -> None:
    arena = load_routing_arena_config(CONFIG_PATH)
    exp = build_arena_experiment_config(arena, dry_run=False, n_layers=4)
    train_t = int(arena["train_context_length"])
    bench = _resolve_synthetic_bench_cfg(exp, train_t)

    assert int(arena["n_layers"]) == 4
    assert bench.task_types == ["pointer_unique"]
    assert bench.scatter_multi_needles is False
    assert bench.synthetic_decoy_keys == 0
    assert int(arena["seed"]) == 45
    assert int(arena["index_pretrain"]["address_index_steps"]) == 10_000
    assert int(arena["learned_address"]["joint_finetune_steps"]) == 20_000
    assert int(arena["key_vector"]["top_k"]) == ROUTING_TOP_K

    la = arena["learned_address"]
    assert la.get("similarity") == "asymmetric"
    assert int(la.get("address_dim", 32)) == 32

    payload = generate_synthetic_task(
        "pointer_unique",
        random.Random(42),
        {"synthetic_decoy_keys": 0, "answer_digit_width": 1},
    )
    assert trace_task_answer(" ".join(payload.needle_segments), payload) == payload.expected_answer

    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(
        context_length=train_t,
        needle_depth=0.5,
        task_type="pointer_unique",
        seed=99,
    )
    assert len(sample.input_ids) <= train_t + 32

    holdout_mid, holdout_full, meta = resolve_holdout_splits(exp, bench, train_t)
    assert len(holdout_full) > 0
    assert meta["holdout_full_samples"] >= 1

    dry = build_arena_experiment_config(arena, dry_run=True, n_layers=4)
    assert int(dry["transformer"]["max_steps"]) <= 100

    print("OK learned_address_proof_cell config")
    print(f"  T={train_t}  variants={PROOF_VARIANTS}")
    print(f"  holdout_official={meta['holdout_full_samples']}")


if __name__ == "__main__":
    main()
