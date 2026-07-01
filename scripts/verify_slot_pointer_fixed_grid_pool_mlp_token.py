#!/usr/bin/env python3
"""Smoke-test pool_mlp_token head wiring for fixed-grid slot_pointer query-only."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer
from experiments.experiment_7 import (
    _batch_uses_query_only_answer,
    _query_only_answer_target_tensor,
    _question_index_batch_tensor,
)
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.generator import LongContextSampleGenerator
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.slot_pointer import (
    _place_fixed_grid_quads,
    verify_slot_pointer_tokens,
)
from routing_attention.models.pointer_head import QueryPoolMLPTokenHead

CFG = ROOT / "configs" / "routing_slot_pointer_fixed_grid_t2048_50q_pool_mlp_token.yaml"


def test_head_shapes() -> None:
    head = QueryPoolMLPTokenHead(
        max_seq_len=2048,
        d_model=256,
        vocab_size=129,
        pool_positions=16,
        n_layers=2,
    )
    b, t, d = 2, 2048, 256
    layers = [torch.randn(b, t, d), torch.randn(b, t, d)]
    q = torch.tensor([t - 1, t - 1], dtype=torch.long)
    out = head(layers, q)
    assert out.shape == (b, 129)


def main() -> None:
    test_head_shapes()

    arena = load_routing_arena_config(CFG)
    exp_cfg = build_arena_experiment_config(arena, dry_run=False, n_layers=2)
    assert exp_cfg["model"]["output_head"] == "pool_mlp_token"
    assert int(exp_cfg["model"]["pool_mlp_positions"]) == 16

    bench = _resolve_synthetic_bench_cfg(exp_cfg, 2048)
    assert bench.slot_quad_placement == "fixed_grid"
    assert bench.train_label_mode == "query_only_answer"
    assert len(_place_fixed_grid_quads(content_length=2047, num_quads=50)) == 50

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
    verify_slot_pointer_tokens(tokens, question_index=q_idx)

    model = build_transformer(exp_cfg, attention_type="dense_flash")
    assert model.n_layers == 2
    assert model.pool_mlp_token_head is not None
    model.eval()

    meta_list = [meta]
    assert _batch_uses_query_only_answer(meta_list)
    q_batch = _question_index_batch_tensor(meta_list, torch.device("cpu"))
    targets = _query_only_answer_target_tensor(meta_list, torch.device("cpu"))
    assert int(targets[0]) == value_tok

    with torch.no_grad():
        out = model(
            input_ids=sample.input_ids.unsqueeze(0),
            attn_mask=None,
            question_index=q_batch,
        )
        token_logits = out["token_logits"]
    assert token_logits.shape == (1, 129)

    evaluator = LongContextEvaluator(bench, holdout_samples=[sample])
    wrong = evaluator.score_sample(token_logits, meta)
    assert not wrong.correct

    perfect = token_logits.clone()
    perfect[0, value_tok] = perfect[0].max() + 10.0
    right = evaluator.score_sample(perfect, meta)
    assert right.correct

    loss = F.cross_entropy(token_logits, targets)
    perfect_loss = F.cross_entropy(perfect, targets)
    assert perfect_loss.item() < 0.01
    assert loss.item() > perfect_loss.item()

    print("ALL pool_mlp_token fixed-grid checks PASSED")


if __name__ == "__main__":
    main()
