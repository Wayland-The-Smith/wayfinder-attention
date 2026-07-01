"""Analyze dense NIAH calibration curves and recommend suite step budgets."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from routing_attention.benchmarks.long_context.generator import LongContextSample


def subsample_holdout_stratified(
    samples: list[LongContextSample],
    *,
    samples_per_cell: int,
    seed: int = 0,
) -> list[LongContextSample]:
    """
    Stratified subsample for mid-train validation (fast but covers all task×depth cells).

    Full holdout is still used for the post-training eval in the calibration script.
    """
    if samples_per_cell <= 0:
        return list(samples)
    by_cell: dict[tuple, list[LongContextSample]] = defaultdict(list)
    for s in samples:
        key = (s.task_type, float(s.needle_depth), str(s.haystack_mode))
        by_cell[key].append(s)
    rng = random.Random(seed)
    out: list[LongContextSample] = []
    for key in sorted(by_cell.keys()):
        cell = list(by_cell[key])
        rng.shuffle(cell)
        out.extend(cell[:samples_per_cell])
    return out


def analyze_holdout_curve(
    mid_validations: list[dict[str, Any]],
    *,
    min_delta_pp: float = 0.005,
    patience_checks: int = 3,
    target_accuracy: float | None = 0.90,
    min_recommended_steps: int = 1000,
) -> dict[str, Any]:
    """
    Turn periodic holdout checks into a recommended step budget for the full suite.

    ``recommended_steps`` is the earliest step whose accuracy is within ``min_delta_pp``
    of the best observed holdout (efficient budget). ``best_steps`` is the step of peak
    holdout accuracy.
    """
    if not mid_validations:
        return {
            "recommended_steps": None,
            "best_steps": None,
            "best_accuracy": None,
            "verdict": "no_validations",
            "reason": "No mid-train holdout evaluations were recorded.",
        }

    def _gate_acc(entry: dict[str, Any]) -> float:
        return float(
            entry.get(
                "primary_gate_accuracy",
                entry.get("pure_niah_accuracy", entry.get("overall_accuracy", 0.0)),
            )
        )

    ordered = sorted(mid_validations, key=lambda v: int(v["step"]))
    best = max(ordered, key=_gate_acc)
    best_acc = _gate_acc(best)
    best_step = int(best["step"])

    threshold = best_acc - min_delta_pp
    efficient_candidates = [
        int(v["step"])
        for v in ordered
        if _gate_acc(v) >= threshold
    ]
    efficient_step = min(efficient_candidates) if efficient_candidates else best_step
    recommended = max(min_recommended_steps, efficient_step)

    plateau_step = None
    stale = 0
    for v in ordered:
        acc = _gate_acc(v)
        if acc > best_acc - min_delta_pp:
            stale = 0
            plateau_step = int(v["step"])
        else:
            stale += 1
            if stale >= patience_checks:
                break

    verdict, reason = _task_verdict(best_acc, target_accuracy)

    return {
        "recommended_steps": recommended,
        "best_steps": best_step,
        "best_accuracy": best_acc,
        "efficient_steps": efficient_step,
        "plateau_near_step": plateau_step,
        "min_delta_pp": min_delta_pp,
        "patience_checks": patience_checks,
        "target_accuracy": target_accuracy,
        "verdict": verdict,
        "reason": reason,
        "holdout_curve": [
            {
                "step": int(v["step"]),
                "primary_gate_accuracy": _gate_acc(v),
                "overall_accuracy": float(v.get("overall_accuracy", 0.0)),
                "correct": int(v.get("correct", 0)),
                "total": int(v.get("total", 0)),
            }
            for v in ordered
        ],
    }


def _task_verdict(best_acc: float, target_accuracy: float | None) -> tuple[str, str]:
    if best_acc < 0.10:
        return (
            "task_broken",
            "Holdout accuracy stayed near chance — check task design, labels, or eval pipeline.",
        )
    if best_acc < 0.30:
        return (
            "weak",
            "Dense learns little retrieval signal — debug task difficulty or training signal before the suite.",
        )
    if best_acc < 0.50:
        return (
            "partial",
            "Some retrieval signal; suite may run but dense ceiling is low.",
        )
    if target_accuracy is not None and best_acc >= target_accuracy:
        return (
            "converged",
            f"Dense reached target holdout accuracy ({target_accuracy:.0%}+).",
        )
    if best_acc >= 0.80:
        return (
            "strong",
            "Dense performs well; use recommended_steps for fair variant comparison.",
        )
    return (
        "moderate",
        "Dense is learning; consider more steps or inspect per-task breakdown.",
    )


def load_calibration_recommendation(path: str | Path) -> dict[str, Any]:
    """Load JSON written by ``scripts/calibrate_dense_niah.py``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rec = data.get("recommendation") or data
    steps = rec.get("recommended_steps")
    if steps is None:
        raise ValueError(f"No recommended_steps in calibration file: {path}")
    return {
        "recommended_steps": int(steps),
        "best_steps": rec.get("best_steps"),
        "best_accuracy": rec.get("best_accuracy"),
        "verdict": rec.get("verdict"),
        "source": str(path),
        "train_context_length": data.get("train_context_length"),
    }


def apply_steps_to_config(config: dict[str, Any], steps: int) -> dict[str, Any]:
    """Return a shallow override dict for suite / experiment runners."""
    steps = int(steps)
    return {
        "transformer": {
            "max_steps": steps,
            "dense_pretrain_steps": steps,
            "sparse_finetune_steps": steps,
        }
    }
