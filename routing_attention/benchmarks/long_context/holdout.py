"""Fixed held-out eval grid shared across all Experiment 7 variants."""

from __future__ import annotations

from typing import Any

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dense_calibration import subsample_holdout_stratified
from routing_attention.benchmarks.long_context.generator import LongContextSample, LongContextSampleGenerator

_GRID_CACHE: dict[tuple, list[LongContextSample]] = {}


def count_holdout_grid_cells(
    bench_cfg: LongContextBenchmarkConfig,
    *,
    context_length: int | None = None,
) -> int:
    """Stratified cells in the fixed holdout grid (task × depth × haystack × T)."""
    tasks = bench_cfg.eval_task_types()
    depths = len(bench_cfg.needle_depths)
    modes = len(bench_cfg.haystack_modes)
    if context_length is not None:
        lengths = 1
    else:
        lengths = len(bench_cfg.context_lengths)
    return len(tasks) * depths * modes * lengths


def resolve_eval_samples_per_cell(
    bench_cfg: LongContextBenchmarkConfig,
    *,
    holdout_total_samples: int | None = None,
    context_length: int | None = None,
) -> int:
    """Map a target holdout size to per-cell sample count (even split across cells)."""
    if holdout_total_samples is None or holdout_total_samples <= 0:
        return int(bench_cfg.eval_samples_per_cell)
    cells = count_holdout_grid_cells(bench_cfg, context_length=context_length)
    if cells <= 0:
        return int(bench_cfg.eval_samples_per_cell)
    return max(1, int(holdout_total_samples) // cells)


def apply_holdout_total_samples(
    bench_cfg: LongContextBenchmarkConfig,
    holdout_total_samples: int,
    *,
    context_length: int | None = None,
) -> LongContextBenchmarkConfig:
    """Return bench config with ``eval_samples_per_cell`` sized for ``holdout_total_samples``."""
    per_cell = resolve_eval_samples_per_cell(
        bench_cfg,
        holdout_total_samples=holdout_total_samples,
        context_length=context_length,
    )
    return LongContextBenchmarkConfig.from_dict(
        {**bench_cfg.to_dict(), "eval_samples_per_cell": per_cell}
    )


def _holdout_total_from_config(config: dict[str, Any]) -> int | None:
    holdout_cfg = config.get("holdout", {}) or {}
    cal_cfg = config.get("dense_calibration", {}) or {}
    total = holdout_cfg.get("total_samples")
    if total is None:
        total = cal_cfg.get("holdout_total_samples")
    if total is None:
        return None
    total = int(total)
    return total if total > 0 else None


def _mid_train_samples_per_cell(config: dict[str, Any]) -> int:
    holdout_cfg = config.get("holdout", {}) or {}
    cal_cfg = config.get("dense_calibration", {}) or {}
    return int(
        holdout_cfg.get("mid_train_samples_per_cell")
        or cal_cfg.get("mid_train_samples_per_cell", 2)
    )


def resolve_holdout_splits(
    config: dict[str, Any],
    bench_cfg: LongContextBenchmarkConfig,
    train_context_length: int,
    *,
    mid_train_seed_offset: int = 0,
) -> tuple[list[LongContextSample], list[LongContextSample], dict[str, Any]]:
    """
    Build standardized holdout splits for training vs official eval.

    Training stream uses ``bench_cfg.seed`` only; holdout uses ``holdout_seed`` (disjoint).
    Mid-train validation uses a stratified subsample; official results use the full grid.
    """
    total = _holdout_total_from_config(config)
    bench_for_holdout = bench_cfg
    if total is not None:
        bench_for_holdout = apply_holdout_total_samples(
            bench_cfg,
            total,
            context_length=train_context_length,
        )

    clear_holdout_cache()
    holdout_all = get_holdout_grid(bench_for_holdout)
    holdout_full = filter_holdout_by_context_length(holdout_all, train_context_length)
    clear_holdout_cache()

    if not holdout_full:
        raise RuntimeError(f"No holdout samples for context_length={train_context_length}")

    per_cell = _mid_train_samples_per_cell(config)
    cal_cfg = config.get("dense_calibration", {}) or {}
    if bool(cal_cfg.get("eval_use_full_holdout", False)):
        holdout_mid = holdout_full
    else:
        holdout_mid = subsample_holdout_stratified(
            holdout_full,
            samples_per_cell=per_cell,
            seed=int(bench_cfg.holdout_seed) + train_context_length + int(mid_train_seed_offset),
        )

    cells = count_holdout_grid_cells(bench_for_holdout, context_length=train_context_length)
    meta: dict[str, Any] = {
        "holdout_total_target": total,
        "holdout_full_samples": len(holdout_full),
        "holdout_mid_samples": len(holdout_mid),
        "holdout_grid_cells": cells,
        "eval_samples_per_cell": bench_for_holdout.eval_samples_per_cell,
        "mid_train_samples_per_cell": per_cell,
        "holdout_seed": bench_cfg.holdout_seed,
        "train_seed": bench_cfg.seed,
        "train_disjoint_from_holdout": bench_cfg.seed != bench_cfg.holdout_seed,
    }
    return holdout_mid, holdout_full, meta


def get_holdout_grid(config: LongContextBenchmarkConfig) -> list[LongContextSample]:
    """
    Build or return cached held-out samples.

    Uses ``holdout_seed`` — disjoint from the training stream (``seed``).
    """
    key = config.cache_key()
    if key not in _GRID_CACHE:
        holdout_cfg = config.holdout_config()
        gen = LongContextSampleGenerator(holdout_cfg)
        _GRID_CACHE[key] = gen.generate_grid(
            task_types=config.eval_task_types(),
            samples_per_cell=holdout_cfg.eval_samples_per_cell,
            num_workers=holdout_cfg.eval_grid_workers,
        )
    return _GRID_CACHE[key]


def clear_holdout_cache() -> None:
    _GRID_CACHE.clear()


def filter_holdout_by_context_length(
    samples: list[LongContextSample],
    context_length: int,
) -> list[LongContextSample]:
    """Held-out samples for a single context-length sub-experiment."""
    return [s for s in samples if s.context_length == context_length]
