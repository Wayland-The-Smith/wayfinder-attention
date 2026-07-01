#!/usr/bin/env python3
"""Diagnose exact gate accuracies (e.g. 25%) on the L2 holdout grid."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import build_transformer, init_experiment_runtime, load_experiment_config, set_seed
from experiments.experiment_4 import _expand_position_embeddings
from experiments.experiment_7 import _benchmark_config_from_run, _load_state_dict_tolerant
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.holdout import clear_holdout_cache, filter_holdout_by_context_length, get_holdout_grid
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile

def _bench_cfg() -> LongContextBenchmarkConfig:
    cfg = LongContextBenchmarkConfig(
        context_lengths=[2048],
        needle_depths=[0.10, 0.25, 0.50, 0.75, 0.90],
        task_types=["pointer_unique"],
        suffix_placement="at_end",
        scatter_multi_needles=False,
        synthetic_decoy_keys=0,
        eval_samples_per_cell=4,
        holdout_seed=1000042,
        benchmark_family="synthetic",
    ).apply_synthetic_profile()
    return cfg


def _possible_accuracies(n: int) -> list[float]:
    return [i / n for i in range(n + 1)]


def _random_digit_baseline(n_trials: int = 10_000, n_samples: int = 20) -> dict:
    """Chance accuracy when guessing one digit (0-9) per sample."""
    hits = []
    for _ in range(n_trials):
        correct = sum(1 for _ in range(n_samples) if random.randint(0, 9) == random.randint(0, 9))
        hits.append(correct / n_samples)
    c = Counter(round(x, 4) for x in hits)
    p_25 = sum(1 for x in hits if x == 0.25) / n_trials
    return {
        "n_trials": n_trials,
        "n_samples": n_samples,
        "mean_accuracy": sum(hits) / len(hits),
        "p_exactly_25pct": p_25,
        "top_discrete_accuracies": c.most_common(8),
    }


def _eval_model(model, bench_cfg, holdout, device) -> list:
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout)
    summary = evaluator.evaluate_module(model, device=device, show_progress=False)
    return summary.records


def main() -> None:
    bench_cfg = _bench_cfg()
    clear_holdout_cache()
    holdout = filter_holdout_by_context_length(get_holdout_grid(bench_cfg), 2048)
    clear_holdout_cache()

    n = len(holdout)
    depths = sorted({float(s.needle_depth) for s in holdout})
    per_cell = bench_cfg.eval_samples_per_cell

    print("=== Holdout grid structure ===")
    print(f"  samples: {n}")
    print(f"  depths: {depths} ({len(depths)} cells)")
    print(f"  samples_per_cell: {per_cell}")
    print(f"  possible overall accuracies (k/{n}): {[f'{a:.0%}' for a in _possible_accuracies(n) if a <= 0.5]}")
    print(f"  possible per-cell accuracies (k/{per_cell}): {[f'{a:.0%}' for a in _possible_accuracies(per_cell)]}")
    print()

    print("=== Random digit baseline (NOT 4-way) ===")
    rb = _random_digit_baseline()
    print(f"  expected mean: {rb['mean_accuracy']:.1%}")
    print(f"  P(exactly 25% on 20 samples): {rb['p_exactly_25pct']:.2%}")
    print(f"  most common discrete accuracies: {rb['top_discrete_accuracies']}")
    print()

    by_depth_meta: dict[float, list[dict]] = defaultdict(list)
    for s in holdout:
        by_depth_meta[float(s.needle_depth)].append(s.metadata)

    print("=== Holdout answer / key distribution ===")
    all_values = [str(s.metadata["value"]) for s in holdout]
    all_keys = [str(s.metadata["key"]) for s in holdout]
    print(f"  unique values: {sorted(set(all_values))} (count={len(set(all_values))})")
    print(f"  value counts: {dict(Counter(all_values))}")
    print(f"  key counts: {dict(Counter(all_keys))}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = load_experiment_config(7, variant="dense_flash")
    config = apply_suite_profile(raw, "fast")
    config["model"]["n_layers"] = 2
    config["model"]["max_seq_len"] = 2048
    config["long_context_benchmark"] = bench_cfg.to_dict()
    init_experiment_runtime(config)
    set_seed(42)

    print("=== Untrained 2L model ===")
    model = build_transformer(config, attention_type="dense_flash").to(device)
    _expand_position_embeddings(model, 2048)
    model.eval()
    records = _eval_model(model, bench_cfg, holdout, device)
    _print_records("untrained", records)

    ckpt = ROOT / "experiments/Experiment_7/feasibility_ladder_2L/checkpoints/level2_T2048_dense_flash.pt"
    if ckpt.exists():
        print(f"=== Trained 2L checkpoint ({ckpt.name}) ===")
        model2 = build_transformer(config, attention_type="dense_flash").to(device)
        _load_state_dict_tolerant(model2, ckpt, device)
        _expand_position_embeddings(model2, 2048)
        model2.eval()
        records2 = _eval_model(model2, bench_cfg, holdout, device)
        _print_records("trained@best", records2)

        pred_counter = Counter(r.predicted.strip() for r in records2)
        exp_counter = Counter(r.expected.strip() for r in records2)
        print(f"  prediction histogram: {dict(pred_counter)}")
        print(f"  expected histogram:   {dict(exp_counter)}")
        print(f"  constant-prediction acc if always '{pred_counter.most_common(1)[0][0]}': "
              f"{sum(1 for r in records2 if r.predicted.strip() == pred_counter.most_common(1)[0][0]) / len(records2):.0%}")
    else:
        print(f"(checkpoint not found: {ckpt})")

    out = ROOT / "experiments/Experiment_7/feasibility_ladder_2L/gate_accuracy_diagnostic.json"
    payload = {
        "holdout_n": n,
        "samples_per_cell": per_cell,
        "n_depth_cells": len(depths),
        "random_digit_baseline": rb,
        "value_counts": dict(Counter(all_values)),
        "key_counts": dict(Counter(all_keys)),
    }
    if ckpt.exists():
        payload["trained"] = {
            "correct": sum(1 for r in records2 if r.correct),
            "total": len(records2),
            "by_depth": _by_depth(records2),
            "predictions": [
                {
                    "depth": r.needle_depth,
                    "expected": r.expected,
                    "predicted": r.predicted,
                    "correct": r.correct,
                }
                for r in records2
            ],
        }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


def _by_depth(records) -> dict:
    bucket: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        bucket[str(r.needle_depth)].append(r.correct)
    return {k: sum(v) / len(v) for k, v in sorted(bucket.items())}


def _print_records(label: str, records) -> None:
    correct = sum(1 for r in records if r.correct)
    total = len(records)
    print(f"  {label}: {correct}/{total} = {correct/total:.0%}")
    print(f"  by_depth: {_by_depth(records)}")
    for r in records:
        mark = "OK" if r.correct else "XX"
        print(f"    [{mark}] d={r.needle_depth:.3f}  exp={r.expected!r}  pred={r.predicted!r}")
    print()


if __name__ == "__main__":
    main()
