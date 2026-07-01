"""Experiment 7 suite profiles — trade wall-clock time vs coverage."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

PROFILE_NAMES = ("fast", "full")


def get_suite_profiles(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return config.get("suite_profiles", {})


def resolve_profile_name(config: dict[str, Any], cli_profile: str | None) -> str:
    if cli_profile and cli_profile in PROFILE_NAMES:
        return cli_profile
    default = config.get("suite", {}).get("default_profile", "fast")
    return default if default in PROFILE_NAMES else "fast"


def apply_suite_profile(config: dict[str, Any], profile_name: str) -> dict[str, Any]:
    """Merge profile overrides into a copy of the experiment config."""
    profiles = get_suite_profiles(config)
    if profile_name not in profiles:
        raise ValueError(f"Unknown suite profile {profile_name!r}; choose from {PROFILE_NAMES}")

    out = deepcopy(config)
    patch = deepcopy(profiles[profile_name])

    for key in (
        "model",
        "transformer",
        "data",
        "evaluation",
        "long_context_benchmark",
        "index_pretrain",
        "router",
        "data_collection",
    ):
        if key in patch:
            out.setdefault(key, {})
            out[key].update(patch.pop(key))

    # Remaining keys (fair_comparison, description, etc.)
    out.setdefault("suite_active_profile", {})
    out["suite_active_profile"] = {
        "name": profile_name,
        **patch,
    }
    return out


def fair_comparison_enabled(profile_meta: dict[str, Any]) -> bool:
    """When true, every variant trains+evals at every T; dense Stage A runs at every T."""
    return bool(profile_meta.get("fair_comparison", True))


def dense_flash_train_skipped(profile_meta: dict[str, Any], train_t: int, variant: str) -> bool:
    """True when dense_flash training/eval is skipped at this T (non-fair profiles only)."""
    if fair_comparison_enabled(profile_meta):
        return False
    if variant != "dense_flash":
        return False
    cap = profile_meta.get("dense_flash_max_train_context")
    if cap is None:
        return False
    return train_t > int(cap)


def variant_run_mode(
    profile_meta: dict[str, Any],
    train_t: int,
    variant: str,
) -> str:
    """
    ``full`` — train + holdout eval + latency (default for fair comparison).
    ``latency_only`` — forward timing only (legacy, non-fair profiles).
    ``skip`` — do not run.
    """
    if fair_comparison_enabled(profile_meta):
        return "full"
    if dense_flash_train_skipped(profile_meta, train_t, variant):
        mode = profile_meta.get("dense_flash_above_cap_mode", "latency_only")
        return mode if mode in ("latency_only", "skip") else "latency_only"
    return "full"


def estimate_run_count(context_lengths: list[int], variants: list[str], profile_meta: dict) -> int:
    n = 0
    for t in context_lengths:
        for v in variants:
            if variant_run_mode(profile_meta, t, v) == "skip":
                continue
            n += 1
    # Stage A dense pretrain once per T (ascending pre-pass)
    if fair_comparison_enabled(profile_meta):
        n += len(context_lengths)
    return n


def dense_checkpoint_path(checkpoint_dir: Path, train_t: int) -> Path:
    return checkpoint_dir / f"T{train_t}_dense_flash.pt"


def dense_pretrain_max_context(profile_meta: dict[str, Any]) -> int | None:
    """Max T for Stage A dense pretrain. None = train dense at every context length."""
    if fair_comparison_enabled(profile_meta):
        return None
    val = profile_meta.get("dense_pretrain_max_context")
    if val is not None:
        return int(val)
    cap = profile_meta.get("dense_flash_max_train_context")
    return int(cap) if cap is not None else None


def resolve_dense_init_checkpoint(
    checkpoint_dir: Path,
    train_t: int,
    profile_meta: dict[str, Any],
    *,
    context_lengths: list[int] | None = None,
) -> Path:
    """
    Return path to C_dense for sparse fine-tune at ``train_t``.

    Fair mode: exact ``T{train_t}_dense_flash.pt`` only (no cross-T fallback).
    """
    primary = dense_checkpoint_path(checkpoint_dir, train_t)
    if fair_comparison_enabled(profile_meta):
        return primary

    if primary.exists():
        return primary

    cap = dense_pretrain_max_context(profile_meta)
    if cap is None or train_t <= cap:
        return primary

    fallback = dense_checkpoint_path(checkpoint_dir, cap)
    if fallback.exists():
        return fallback

    if context_lengths:
        for t in sorted(context_lengths, reverse=True):
            if cap is not None and t > cap:
                continue
            candidate = dense_checkpoint_path(checkpoint_dir, t)
            if candidate.exists():
                return candidate
    return primary


def needs_dense_checkpoint(
    variants: list[str],
    profile_meta: dict,
    train_t: int,
    variant: str,
) -> bool:
    """True when this variant run requires a dense checkpoint for two-stage init."""
    if variant == "dense_flash":
        return False
    if variant_run_mode(profile_meta, train_t, variant) != "full":
        return False
    return True
