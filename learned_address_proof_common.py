"""Shared helpers for learned-address proof / breakthrough experiment suites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent

CONFIG_PATH = ROOT / "configs" / "learned_address_proof_cell" / "niah_pointer_unique_t2048_4L.yaml"
HARD_CELL_CONFIG_PATH = ROOT / "configs" / "learned_address_proof_cell" / "niah_pointer_unique_t2048_4L_decoy1.yaml"

OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "learned_address_proof_cell"
BREAKTHROUGH_OUTPUT = ROOT / "experiments" / "Experiment_7" / "learned_address_breakthrough"

PROOF_CELL_DENSE_CKPT = OUTPUT_ROOT / "proof_cell_full" / "checkpoints" / "dense_flash.pt"

CANONICAL_SEED = 45
SEED_REPRO_SEEDS = [43, 45, 46]
ROUTING_TOP_K = 128
PHASE_B_STEPS = 10_000
PHASE_C_STEPS = 20_000
DENSE_STEPS = 20_000
CACHE_TRAIN_BATCHES = 64
CACHE_HOLDOUT_BATCHES = 16

SWEEP_B_STEPS = [2_000, 5_000, 10_000, 20_000]
CURRICULUM_LENGTHS = [2048, 4096, 8192]

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

PHASE_C_VARIANTS = (
    "linear",
    "key_vector_k32",
    "learned_address_k32",
    "local_window64",
    "local_window256",
)

PROOF_VARIANTS = ("dense_flash", *PHASE_C_VARIANTS)

SYSTEMS_VARIANTS = (
    "dense_flash",
    "linear",
    "key_vector_k32",
    "learned_address_k32",
)
SYSTEMS_LENGTHS = [2048, 4096, 8192, 16384]

BREAKTHROUGH_CLAIM = (
    "Learned-address sparse attention matches dense retrieval on NIAH, beats key-vector "
    "and fixed local sparse, and reduces long-context latency ~2× vs dense — with a "
    "3-phase training protocol (dense teacher → address index → sparse finetune)."
)


def resolve_dense_checkpoint(explicit: Path | None = None) -> Path | None:
    if explicit and explicit.exists():
        return explicit
    if PROOF_CELL_DENSE_CKPT.exists():
        return PROOF_CELL_DENSE_CKPT
    if DEFAULT_DENSE_CKPT.exists():
        return DEFAULT_DENSE_CKPT
    return None


def official_accuracy(payload: dict) -> float | None:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    return float(acc) if acc is not None else None


def depth_stratified(payload: dict) -> dict[str, float]:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    raw = ev.get("by_needle_depth") or {}
    return {str(k): float(v) for k, v in raw.items()}


def recall_at_k(metrics: dict[str, Any], k: int) -> float | None:
    if not metrics:
        return None
    key = f"recall@{min(k, metrics.get('seq_len', k))}"
    if key in metrics:
        return float(metrics[key])
    mean = metrics.get("mean_recall")
    if mean is not None:
        return float(mean)
    per_layer = metrics.get("per_layer") or {}
    if not per_layer:
        return None
    vals = []
    for layer_metrics in per_layer.values():
        for mk, mv in layer_metrics.items():
            if mk.startswith("recall@"):
                vals.append(float(mv))
                break
    return sum(vals) / len(vals) if vals else None


def post_phase_c_recall(payload: dict, recall_k: int) -> float | None:
    post = payload.get("post_phase_c_recall") or {}
    return recall_at_k(post, recall_k)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


@torch.inference_mode()
def measure_address_recall(
    address_book: nn.Module,
    holdout_cache: Path | str,
    *,
    device: torch.device,
    recall_k: int,
    n_layers: int,
    dry_run: bool,
) -> dict[str, Any]:
    from routing_attention.evaluation.recall import evaluate_learned_address_recall_from_cache

    max_tokens = 128 if dry_run else 0
    return evaluate_learned_address_recall_from_cache(
        address_book,
        holdout_cache,
        device,
        recall_k=recall_k,
        n_layers=n_layers,
        max_eval_tokens=max_tokens,
        show_progress=not dry_run,
    )


@torch.inference_mode()
def measure_key_vector_recall(
    model: nn.Module,
    holdout_cache: Path | str,
    *,
    device: torch.device,
    recall_k: int,
    n_layers: int,
    dry_run: bool,
) -> dict[str, Any]:
    from routing_attention.evaluation.recall import evaluate_key_vector_recall_from_cache

    max_tokens = 128 if dry_run else 0
    return evaluate_key_vector_recall_from_cache(
        model,
        holdout_cache,
        device,
        recall_k=recall_k,
        n_layers=n_layers,
        max_eval_tokens=max_tokens,
        show_progress=not dry_run,
    )
