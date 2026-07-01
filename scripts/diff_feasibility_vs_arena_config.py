#!/usr/bin/env python3
"""Dump resolved feasibility vs arena configs for harness drift debugging."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.feasibility_ladder import (
    _merge_level_config,
    _resolve_bench_cfg as feasibility_resolve_bench,
    load_feasibility_ladder_config,
)
from routing_attention.benchmarks.long_context.holdout import resolve_holdout_splits
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    load_routing_arena_config,
)

FEASIBILITY_CFG = ROOT / "configs" / "feasibility_ladder_4L_20k.yaml"
ARENA_CFG = ROOT / "configs" / "harness_dense_parity" / "niah_4L_t2048.yaml"
OUT = ROOT / "experiments" / "Experiment_7" / "harness_diff"


def _pick(d: dict, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        parts = k.split(".")
        cur = d
        ok = True
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                ok = False
                break
            cur = cur[p]
        if ok:
            out[k] = cur
    return out


def main() -> None:
    ladder_raw = yaml.safe_load(FEASIBILITY_CFG.read_text(encoding="utf-8"))
    ladder_cfg = ladder_raw.get("feasibility_ladder", ladder_raw)
    level = next(l for l in ladder_cfg["levels"] if l["level"] == 2)

    feas_config = _merge_level_config(ladder_cfg, level, dry_run=False)
    feas_bench = feasibility_resolve_bench(feas_config, level)
    feas_mid, feas_full, feas_meta = resolve_holdout_splits(
        feas_config, feas_bench, 2048, mid_train_seed_offset=2
    )

    arena_cfg = load_routing_arena_config(ARENA_CFG)
    arena_config = build_arena_experiment_config(arena_cfg, dry_run=False, n_layers=4)
    arena_bench = _resolve_synthetic_bench_cfg(arena_config, 2048)
    arena_mid, arena_full, arena_meta = resolve_holdout_splits(
        arena_config, arena_bench, 2048, mid_train_seed_offset=7
    )

    keys = [
        "model.n_layers",
        "model.d_model",
        "model.n_heads",
        "model.dropout",
        "model.vocab_size",
        "model.attention_type",
        "transformer.max_steps",
        "transformer.dense_pretrain_steps",
        "transformer.sparse_finetune_steps",
        "transformer.lr",
        "transformer.validate_every",
        "data.batch_size",
        "dense_calibration.restore_best_checkpoint",
        "dense_calibration.eval_use_full_holdout",
        "dense_calibration.mid_train_samples_per_cell",
        "holdout.total_samples",
        "routing_attention.fair_finetune",
        "long_context_benchmark.question_prefix",
        "long_context_benchmark.answer_prefix",
        "long_context_benchmark.train_label_mode",
        "long_context_benchmark.include_answer_in_suffix",
        "long_context_benchmark.answer_loss_weight",
        "long_context_benchmark.scatter_multi_needles",
        "long_context_benchmark.synthetic_decoy_keys",
        "long_context_benchmark.eval_samples_per_cell",
        "long_context_benchmark.haystack_modes",
        "long_context_benchmark.generation_max_attempts",
        "long_context_benchmark.benchmark_version",
    ]

    feas_pick = _pick(feas_config, keys)
    arena_pick = _pick(arena_config, keys)

    bench_keys = [
        "task_types",
        "question_prefix",
        "answer_prefix",
        "train_label_mode",
        "include_answer_in_suffix",
        "answer_loss_weight",
        "scatter_multi_needles",
        "synthetic_decoy_keys",
        "eval_samples_per_cell",
        "haystack_modes",
        "vocab_size",
        "suffix_placement",
        "benchmark_version",
    ]

    diff = {}
    for k in keys:
        fv, av = feas_pick.get(k), arena_pick.get(k)
        if fv != av:
            diff[k] = {"feasibility": fv, "arena": av}

    report = {
        "feasibility_resolved": feas_pick,
        "arena_resolved": arena_pick,
        "diff": diff,
        "feasibility_bench": {k: getattr(feas_bench, k) for k in bench_keys},
        "arena_bench": {k: getattr(arena_bench, k) for k in bench_keys},
        "feasibility_holdout_meta": feas_meta,
        "arena_holdout_meta": arena_meta,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "config_diff.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== feasibility vs arena config diff ===")
    print(f"  wrote: {out_path}")
    print(f"  differing keys: {len(diff)}")
    for k, v in sorted(diff.items()):
        print(f"  {k}:")
        print(f"    feasibility: {v['feasibility']}")
        print(f"    arena:       {v['arena']}")
    print(f"  feas holdout: mid={feas_meta['holdout_mid_samples']} full={feas_meta['holdout_full_samples']} per_cell={feas_meta['eval_samples_per_cell']}")
    print(f"  arena holdout: mid={arena_meta['holdout_mid_samples']} full={arena_meta['holdout_full_samples']} per_cell={arena_meta['eval_samples_per_cell']}")


if __name__ == "__main__":
    main()
