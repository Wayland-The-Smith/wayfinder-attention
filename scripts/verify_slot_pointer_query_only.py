#!/usr/bin/env python3
"""Smoke-test slot_pointer query-only value token generation and eval wiring."""

from __future__ import annotations

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
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)
from routing_attention.benchmarks.long_context.slot_pointer import (
    trace_value_index,
    verify_slot_pointer_tokens,
)


def main() -> None:
    cfg_path = ROOT / "configs" / "routing_slot_pointer_t2048_50q_query_only.yaml"
    arena = load_routing_arena_config(cfg_path)
    exp_cfg = build_arena_experiment_config(arena, dry_run=False)
    assert exp_cfg["model"]["output_head"] == "lm_token"
    bench = _resolve_synthetic_bench_cfg(exp_cfg, 2048)
    assert bench.train_label_mode == "query_only_answer"

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
    target_idx = int(meta["pointer_target_index"])
    value_tok = int(meta["pointer_target_token"])

    assert meta["label_mode"] == "query_only_answer"
    assert meta["answer_supervision"] == "token_id"
    assert meta["metadata"]["protocol"] == "slot_pointer_query_only_v1"
    assert tokens[q_idx] == int(meta["query_addr_token"])
    assert int(sample.labels[q_idx].item()) == value_tok
    assert float(sample.loss_weights_np[q_idx]) > 0
    traced = trace_value_index(tokens, question_index=q_idx)
    assert traced == target_idx
    assert tokens[traced] == value_tok
    verify_slot_pointer_tokens(
        tokens,
        question_index=q_idx,
        expected_value_index=target_idx,
    )

    evaluator = LongContextEvaluator(bench, holdout_samples=[sample])
    device = torch.device("cpu")
    model = build_transformer(exp_cfg, attention_type="dense_flash").to(device)
    assert getattr(model, "output_head", "") == "lm_token"

    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=sample.input_ids.unsqueeze(0).to(device),
            attn_mask=None,
        )
        logits = out["logits"] if isinstance(out, dict) else out

    wrong = evaluator.score_sample(logits, meta)
    assert not wrong.correct

    batch, seq, vocab = logits.shape
    logits[0, -1, value_tok] = logits[0, -1].max() + 10.0
    right = evaluator.score_sample(logits, meta)
    assert right.correct, f"expected hit on value token {value_tok}, got {right.predicted}"

    print("ALL slot_pointer query-only checks PASSED")


if __name__ == "__main__":
    print("=== verify slot_pointer query-only ===")
    main()
