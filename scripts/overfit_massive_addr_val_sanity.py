#!/usr/bin/env python3
"""Overfit sanity: massive_addr_val N=2, 32 fixed samples, needles in chars 0-200."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.feasibility_ladder import run_dense_level
from routing_attention.benchmarks.long_context.runtime import collect_device_info
from routing_attention.models.fast_attention import backend_status

level = {
    "level": 0,
    "name": "massive_addr_val_overfit",
    "train_context_length": 1024,
    "save_checkpoint": False,
    "pass_criteria": {"primary_gate_accuracy_min": 0.95},
    "long_context_benchmark": {
        "benchmark_family": "synthetic",
        "context_lengths": [1024],
        "task_types": ["massive_addr_val"],
        "needle_depths": [0.5],
        "suffix_placement": "at_end",
        "scatter_multi_needles": True,
        "scatter_placement_min": 0,
        "scatter_placement_max": 200,
        "num_kv_pairs": 2,
        "num_distractors": 0,
        "overfit_train_samples": 32,
        "overfit_eval_same_samples": True,
        "eval_samples_per_cell": 32,
        "min_haystack_side_chars": 8,
    },
    "transformer": {
        "max_steps": 800,
        "dense_pretrain_steps": 800,
        "validate_every": 50,
        "log_every": 100,
    },
    "dense_calibration": {
        "live_metrics": True,
        "restore_best_checkpoint": True,
        "mid_train_samples_per_cell": 32,
    },
}

if __name__ == "__main__":
    device = torch.device("cuda")
    out = ROOT / "experiments" / "Experiment_7" / "massive_addr_val_overfit_sanity"
    out.mkdir(parents=True, exist_ok=True)
    r = run_dense_level(
        {"suite_profile": "full", "synthetic": True, "holdout": {}, "levels": [level]},
        level,
        dry_run=False,
        output_dir=out,
        device_info={**collect_device_info(device), **backend_status(), "dry_run": False},
        dense_checkpoint=None,
    )
    acc = r.get("eval", {}).get("primary_gate_accuracy")
    print(f"FINAL_ACC={acc} passed={r.get('pass_result', {}).get('passed')}")
