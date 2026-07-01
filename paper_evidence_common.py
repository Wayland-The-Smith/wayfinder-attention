"""Shared config helpers for paper-evidence experiment suites."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

DEFAULT_DENSE_CKPT = (
    ROOT
    / "experiments"
    / "Experiment_7"
    / "dense_stability_sweep"
    / "replicates"
    / "seed_45"
    / "rep_1"
    / "dense_flash"
    / "dense_flash.pt"
)

PAPER_OUTPUT = ROOT / "experiments" / "Experiment_7" / "paper_evidence"

CANONICAL_SEED = 45
SEED_REPRO_SEEDS = [43, 45, 46]
ROUTING_TOP_K = 128
ROUTING_LR = 3e-4

LENGTH_LEVELS_L2S = [16384, 8192, 4096, 2048]
SYSTEMS_LENGTHS = [2048, 4096, 8192, 16384]

HEADTOHEAD_VARIANTS = (
    "dense_flash",
    "linear",
    "key_vector_k32",
    "local_window64",
    "local_window256",
)

# Skip length-scaling training when a single step exceeds this (ms).
MAX_TRAIN_STEP_MS = 15_000.0
# Skip when projected 10k-step run exceeds this many hours (per variant).
MAX_TRAIN_HOURS_10K = 12.0


def _bench_patch(*, train_t: int, decoys: int, tag: str) -> dict[str, Any]:
    return {
        "benchmark_family": "synthetic",
        "context_lengths": [train_t],
        "task_types": ["pointer_unique"],
        "needle_depths": [0.10, 0.25, 0.50, 0.75, 0.90],
        "suffix_placement": "at_end",
        "scatter_multi_needles": False,
        "synthetic_decoy_keys": decoys,
        "num_distractors": decoys,
        "eval_samples_per_cell": 4,
        "generation_max_attempts": 16,
        "answer_digit_width": 1,
        "answer_loss_weight": 8.0,
        "train_label_mode": "answer_only",
        "include_answer_in_suffix": True,
        "synthetic_hop_count": 1,
        "synthetic_hop_count_min": 1,
        "synthetic_hop_count_max": 1,
        "benchmark_variant": tag,
        "training_protocol": "two_stage",
    }


def arena_base(*, n_layers: int = 4, seed: int = CANONICAL_SEED) -> dict[str, Any]:
    return {
        "description": "paper evidence 4L pointer_unique stable recipe",
        "feasibility_parity": True,
        "holdout_mid_seed_offset": 2,
        "seed": seed,
        "suite_profile": "fast",
        "n_layers": n_layers,
        "dense_finetune_on_task": True,
        "dense_train_from_scratch": True,
        "dense_gate_min": 0,
        "holdout": {"total_samples": 300, "mid_train_samples_per_cell": 10},
        "dense_checkpoint": None,
        "training": {"cudnn_deterministic": True, "cudnn_benchmark": False},
        "transformer": {
            "max_steps": 20000,
            "dense_pretrain_steps": 20000,
            "sparse_finetune_steps": 10000,
            "validate_every": 500,
            "validate_every_min": 500,
            "log_every": 200,
            "lr": 3.0e-4,
            "lr_warmup_steps": 500,
        },
        "dense_calibration": {
            "live_metrics": True,
            "early_stop": False,
            "restore_best_checkpoint": True,
            "eval_use_full_holdout": True,
            "mid_train_samples_per_cell": 4,
            "target_accuracy": 0.90,
        },
        "routing_attention": {"fair_finetune": True},
        "key_vector": {
            "top_k": ROUTING_TOP_K,
            "sparse_finetune_steps": 10000,
            "sparse_finetune_lr": ROUTING_LR,
        },
        "router": {"top_k": ROUTING_TOP_K},
        "dry_run": {
            "max_steps": 80,
            "sparse_finetune_steps": 80,
            "dense_pretrain_steps": 80,
            "validate_every": 20,
            "log_every": 10,
            "mid_train_samples_per_cell": 4,
        },
    }


def build_cell_config(
    *,
    train_t: int,
    decoys: int,
    dry_run: bool,
    n_layers: int,
    seed: int,
    tag: str,
    train_steps: int | None = None,
) -> dict:
    from routing_attention.benchmarks.long_context.routing_arena import build_arena_experiment_config

    arena = copy.deepcopy(arena_base(n_layers=n_layers, seed=seed))
    arena["train_context_length"] = train_t
    arena["long_context_benchmark"] = _bench_patch(
        train_t=train_t, decoys=decoys, tag=tag
    )
    if train_steps is not None:
        arena["transformer"]["max_steps"] = train_steps
        arena["transformer"]["dense_pretrain_steps"] = train_steps
        arena["transformer"]["sparse_finetune_steps"] = train_steps
        arena["key_vector"]["sparse_finetune_steps"] = train_steps
    return build_arena_experiment_config(arena, dry_run=dry_run, n_layers=n_layers)


def official_accuracy(payload: dict) -> float | None:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    return float(acc) if acc is not None else None


def projected_hours(step_ms: float | None, steps: int = 10_000) -> float | None:
    if step_ms is None or step_ms <= 0:
        return None
    return (step_ms * steps) / 1000.0 / 3600.0


def train_feasible(step_ms: float | None, steps: int = 10_000) -> bool:
    if step_ms is None:
        return False
    if step_ms > MAX_TRAIN_STEP_MS:
        return False
    hours = projected_hours(step_ms, steps)
    return hours is not None and hours <= MAX_TRAIN_HOURS_10K
