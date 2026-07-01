"""Breakthrough / success criteria checks for Experiment 7."""

from __future__ import annotations

from typing import Any


def _mean_accuracy_at_lengths(
    by_context_length: dict[str | int, float],
    min_length: int,
) -> float | None:
    vals = [
        float(v)
        for k, v in by_context_length.items()
        if int(k) >= min_length
    ]
    return sum(vals) / len(vals) if vals else None


def _pp(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return (a - b) * 100.0


def evaluate_success_criteria(
    variant_results: dict[str, dict[str, Any]],
    criteria: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate configured breakthrough criteria from per-variant summary dicts.

    ``variant_results`` maps variant name -> result with at least ``summary`` key.
    """
    c = criteria or {}
    routing_var = c.get("routing_variant", "routing_asymmetric")
    local_var = c.get("local_baseline", "local_window256")
    dense_var = c.get("dense_reference", "dense_flash")
    min_t = int(c.get("min_context_length", 8192))
    beats_local_pp = float(c.get("routing_beats_local_window_pp", 0.15)) * 100.0
    max_dense_gap_pp = float(c.get("max_dense_gap_pp", 0.10)) * 100.0

    def _summary(name: str) -> dict[str, Any]:
        return variant_results.get(name, {}).get("summary", {})

    routing = _summary(routing_var)
    local = _summary(local_var)
    dense = _summary(dense_var)

    routing_long = _mean_accuracy_at_lengths(routing.get("by_context_length", {}), min_t)
    local_long = _mean_accuracy_at_lengths(local.get("by_context_length", {}), min_t)
    dense_long = _mean_accuracy_at_lengths(dense.get("by_context_length", {}), min_t)

    routing_vs_local_pp = _pp(routing_long, local_long)
    routing_vs_dense_pp = _pp(routing_long, dense_long)
    dense_gap_pp = abs(routing_vs_dense_pp) if routing_vs_dense_pp is not None else None

    checks = {
        "routing_beats_local_at_long_context": {
            "pass": routing_vs_local_pp is not None and routing_vs_local_pp >= beats_local_pp,
            "routing_acc": routing_long,
            "local_acc": local_long,
            "margin_pp": routing_vs_local_pp,
            "required_pp": beats_local_pp,
        },
        "routing_near_dense_at_long_context": {
            "pass": (
                routing_long is not None
                and dense_long is not None
                and dense_gap_pp is not None
                and dense_gap_pp <= max_dense_gap_pp
            ),
            "routing_acc": routing_long,
            "dense_acc": dense_long,
            "gap_pp": dense_gap_pp,
            "max_gap_pp": max_dense_gap_pp,
        },
        "min_primary_gate_accuracy": {
            "pass": routing.get("primary_gate_accuracy", routing.get("pure_niah_accuracy", 0.0))
            >= float(c.get("min_primary_gate_accuracy", c.get("min_overall_accuracy", 0.50))),
            "routing_primary_gate": routing.get(
                "primary_gate_accuracy", routing.get("pure_niah_accuracy")
            ),
            "threshold": float(c.get("min_primary_gate_accuracy", c.get("min_overall_accuracy", 0.50))),
        },
        "min_overall_accuracy": {
            "pass": routing.get("overall_accuracy", 0.0) >= float(c.get("min_overall_accuracy", 0.50)),
            "routing_overall": routing.get("overall_accuracy"),
            "threshold": float(c.get("min_overall_accuracy", 0.50)),
        },
    }

    tier = "none"
    if checks["routing_beats_local_at_long_context"]["pass"]:
        tier = "interesting"
    if checks["routing_beats_local_at_long_context"]["pass"] and checks[
        "routing_near_dense_at_long_context"
    ]["pass"]:
        tier = "strong"
    if (
        tier == "strong"
        and checks["min_primary_gate_accuracy"]["pass"]
        and routing_vs_local_pp is not None
        and routing_vs_local_pp >= beats_local_pp
    ):
        tier = "breakthrough_candidate"

    return {
        "tier": tier,
        "checks": checks,
        "routing_variant": routing_var,
        "local_baseline": local_var,
        "dense_reference": dense_var,
        "min_context_length": min_t,
        "all_pass": all(v["pass"] for v in checks.values()),
    }
