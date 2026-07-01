#!/usr/bin/env python3
"""Smoke-test fixed-grid slot_pointer query-only value token generation and eval."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from experiments.experiment_7 import (
    _batch_uses_query_only_answer,
    _query_only_answer_aligned_loss,
    _weighted_long_context_loss,
)
from routing_attention.benchmarks.long_context.slot_pointer import (
    _place_fixed_grid_quads,
    verify_slot_pointer_tokens,
)

CFG = ROOT / "configs" / "routing_slot_pointer_fixed_grid_t2048_50q_query_only.yaml"


def main() -> None:
    arena = load_routing_arena_config(CFG)
    exp_cfg = build_arena_experiment_config(arena, dry_run=False, n_layers=2)
    assert exp_cfg["model"]["output_head"] == "lm_token"
    bench = _resolve_synthetic_bench_cfg(exp_cfg, 2048)
    assert bench.slot_quad_placement == "fixed_grid"
    assert bench.train_label_mode == "query_only_answer"

    starts = _place_fixed_grid_quads(content_length=2047, num_quads=50)
    assert len(starts) == 50 and starts[0] == 36

    gen = LongContextSampleGenerator(bench)
    sample = gen.generate_one(
        context_length=2048,
        needle_depth=0.5,
        task_type="slot_pointer",
        seed=42,
    )
    meta = sample.meta_dict
    tokens = sample.input_ids.tolist()
    q_idx = int(meta["question_index"])
    value_tok = int(meta["pointer_target_token"])

    assert meta["label_mode"] == "query_only_answer"
    assert int(sample.labels[q_idx].item()) == value_tok
    verify_slot_pointer_tokens(tokens, question_index=q_idx)

    evaluator = LongContextEvaluator(bench, holdout_samples=[sample])
    model = build_transformer(exp_cfg, attention_type="dense_flash")
    assert model.n_layers == 2
    model.eval()
    with torch.no_grad():
        out = model(input_ids=sample.input_ids.unsqueeze(0), attn_mask=None)
        logits = out["logits"]

    wrong = evaluator.score_sample(logits, meta)
    assert not wrong.correct
    logits[0, -1, value_tok] = logits[0, -1].max() + 10.0
    right = evaluator.score_sample(logits, meta)
    assert right.correct

    meta_list = [meta]
    assert _batch_uses_query_only_answer(meta_list)
    q_idx = int(meta["question_index"])
    labels = sample.labels.unsqueeze(0)
    weights = sample.loss_weights_np
    loss_weights = torch.from_numpy(weights).unsqueeze(0).float()

    shifted = _weighted_long_context_loss(
        logits[:, :-1, :],
        labels[:, 1:],
        loss_weights[:, 1:],
    )
    aligned = _query_only_answer_aligned_loss(logits, labels, loss_weights, meta_list)
    assert shifted.item() != aligned.item()

    perfect = logits.clone()
    perfect[0, q_idx, value_tok] = perfect[0, q_idx].max() + 10.0
    aligned_perfect = _query_only_answer_aligned_loss(perfect, labels, loss_weights, meta_list)
    assert aligned_perfect.item() < 0.01

    print("ALL fixed-grid query-only checks PASSED")


if __name__ == "__main__":
    main()
