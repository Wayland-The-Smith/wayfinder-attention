#!/usr/bin/env python3
"""
Dense-only NIAH calibration — standalone from the Experiment 7 variant suite.

Trains ``dense_flash`` at a fixed context length (default T=8192, ``fast`` 2L profile),
records holdout accuracy over training, and recommends ``dense_pretrain_steps`` /
``sparse_finetune_steps`` for the fair comparison suite.

Does not write suite dense checkpoints or run other variants.

Example:
  python scripts/calibrate_dense_niah.py --save-checkpoint
  python scripts/calibrate_dense_niah.py --dry-run

Apply recommendation to the suite:
  python run_experiment_7_suite.py --profile full \\
    --calibration-steps-json experiments/Experiment_7/dense_calibration/latest.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common import init_experiment_runtime, load_experiment_config, set_seed
from experiments.experiment_7 import (
    _build_variant_model,
    _eval_holdout,
    _save_dense_checkpoint,
    _train_on_benchmark,
    _verify_staged_training_protocol,
)
from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.dense_calibration import (
    analyze_holdout_curve,
    subsample_holdout_stratified,
)
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.holdout import (
    clear_holdout_cache,
    filter_holdout_by_context_length,
    get_holdout_grid,
)
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, peak_vram_mb, reset_peak_vram
from routing_attention.benchmarks.long_context.suite_profile import apply_suite_profile
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.config import load_config, merge_configs
from routing_attention.utils.experiment import get_experiments_root
from routing_attention.utils.logging import setup_logging

logger = logging.getLogger("calibrate_dense_niah")


def _load_calibration_config(
    *,
    profile: str,
    max_steps: int | None,
    validate_every: int | None,
    validate_every_min: int | None,
    log_every: int | None,
    dry_run: bool,
    cal_overrides: dict | None,
) -> dict:
    raw = load_config(ROOT / "configs" / "experiment_7.yaml")
    config = apply_suite_profile(raw, profile)
    cal_defaults = dict(config.get("dense_calibration", {}))

    if dry_run:
        cal_defaults.update(
            {
                "max_steps": 40,
                "validate_every": 10,
                "validate_every_min": 10,
                "log_every": 10,
            }
        )
    if cal_overrides:
        cal_defaults.update(cal_overrides)

    steps_cap = int(max_steps if max_steps is not None else cal_defaults.get("max_steps", 100000))
    transformer_patch: dict = {
        "validate_every": validate_every if validate_every is not None else cal_defaults.get("validate_every", 1000),
        "validate_every_min": validate_every_min
        if validate_every_min is not None
        else cal_defaults.get("validate_every_min", 500),
        "log_every": log_every if log_every is not None else cal_defaults.get("log_every", 5),
        "max_steps": steps_cap,
        "dense_pretrain_steps": steps_cap,
    }

    dense_cal = {
        "min_delta_pp": float(cal_defaults.get("min_delta_pp", 0.005)),
        "target_accuracy": cal_defaults.get("target_accuracy", 0.90),
        "min_recommended_steps": int(cal_defaults.get("min_recommended_steps", 1000)),
        "live_metrics": bool(cal_defaults.get("live_metrics", True)),
        "mid_train_samples_per_cell": int(cal_defaults.get("mid_train_samples_per_cell", 2)),
    }

    ovr = {
        "model": config.get("model", {}),
        "transformer": {**config.get("transformer", {}), **transformer_patch},
        "data": config.get("data", {}),
        "long_context_benchmark": config.get("long_context_benchmark", {}),
        "dense_calibration": dense_cal,
        "suite_active_profile": config.get("suite_active_profile", {}),
    }
    return merge_configs(load_experiment_config(7, variant="dense_flash"), ovr)


def _preflight() -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    print("=== Dense NIAH calibration preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — calibration will be slow and not representative.")
    if not info.get("fla_linear"):
        print("ERROR: flash-linear-attention required.")
        sys.exit(1)
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print()
    return info


def run_calibration(
    *,
    profile: str,
    train_context_length: int,
    max_steps: int | None,
    validate_every: int | None,
    validate_every_min: int | None,
    log_every: int | None,
    dry_run: bool,
    output_dir: Path | None,
    save_checkpoint: bool,
    synthetic: bool = False,
) -> dict:
    config = _load_calibration_config(
        profile=profile,
        max_steps=max_steps,
        validate_every=validate_every,
        validate_every_min=validate_every_min,
        log_every=log_every,
        dry_run=dry_run,
        cal_overrides=None,
    )

    bench_cfg = LongContextBenchmarkConfig.from_dict(config.get("long_context_benchmark", {}))
    if synthetic:
        bench_cfg = bench_cfg.apply_synthetic_profile()
    if dry_run:
        bench_cfg = bench_cfg.apply_dry_run_profile()

    device = init_experiment_runtime(config)
    set_seed(config.get("seed", 42))

    clear_holdout_cache()
    holdout_all = get_holdout_grid(bench_cfg)
    holdout_full = filter_holdout_by_context_length(holdout_all, train_context_length)
    if not holdout_full:
        raise RuntimeError(f"No holdout samples for T={train_context_length}")

    cal_cfg = config.get("dense_calibration", {})
    per_cell = int(cal_cfg.get("mid_train_samples_per_cell", 2))
    holdout_mid = subsample_holdout_stratified(
        holdout_full,
        samples_per_cell=per_cell,
        seed=int(bench_cfg.holdout_seed) + train_context_length,
    )
    validate_every = int(config.get("transformer", {}).get("validate_every", 1000))
    log_every = int(config.get("transformer", {}).get("log_every", 5))
    train_steps = int(config.get("transformer", {}).get("max_steps", 0))
    n_mid_vals = train_steps // validate_every if validate_every > 0 else 0

    print("=== Calibration schedule ===")
    print(f"  benchmark_family: {bench_cfg.benchmark_family}  version: {bench_cfg.benchmark_version}")
    print(f"  train_context_length: {train_context_length}")
    print(f"  n_layers: {config.get('model', {}).get('n_layers')}")
    print(f"  max_steps: {train_steps}  log_every: {log_every}  validate_every: {validate_every}")
    print(f"  mid-train holdout: {len(holdout_mid)} samples ({per_cell}/cell, stratified)")
    print(f"  final holdout:     {len(holdout_full)} samples (full grid)")
    print(f"  ~{n_mid_vals} mid-train validations (+ final eval at step {train_steps})")
    print()

    max_seq = max(train_context_length, int(config.get("model", {}).get("max_seq_len", train_context_length)))
    reset_peak_vram(device)
    t0 = time.perf_counter()

    model, var_config = _build_variant_model(
        config,
        "dense_flash",
        device,
        max_seq,
    )
    audit = _verify_staged_training_protocol(
        "dense_flash",
        "dense_pretrain",
        getattr(model, "_exp7_routing_info", {}),
        model,
        two_stage=True,
    )

    train_info = _train_on_benchmark(
        model,
        var_config,
        bench_cfg,
        holdout_mid,
        device,
        train_context_length,
        logger,
        max_steps=train_steps,
        training_stage="dense_pretrain",
    )

    print()
    print("=== Final full holdout eval ===")
    evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout_full)
    final_summary = evaluator.evaluate_module(model, device=device, show_progress=True)
    wall_sec = time.perf_counter() - t0
    peak_mb = peak_vram_mb(device)

    recommendation = analyze_holdout_curve(
        train_info.get("mid_validations", []),
        min_delta_pp=float(cal_cfg.get("min_delta_pp", 0.005)),
        patience_checks=int(cal_cfg.get("patience_checks", 3)),
        target_accuracy=cal_cfg.get("target_accuracy"),
        min_recommended_steps=int(cal_cfg.get("min_recommended_steps", 1000)),
    )

    out_dir = output_dir or (get_experiments_root() / "Experiment_7" / "dense_calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "calibration"
    json_path = out_dir / f"dense_{tag}_T{train_context_length}_{stamp}.json"
    latest_path = out_dir / "latest.json"

    checkpoint_path = None
    if save_checkpoint and not dry_run:
        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / f"T{train_context_length}_dense_flash_calibrated.pt"
        _save_dense_checkpoint(
            model,
            checkpoint_path,
            train_context_length=train_context_length,
            trained_steps=int(train_info.get("trained_steps", 0)),
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "dense_niah_calibration",
        "benchmark_family": bench_cfg.benchmark_family,
        "benchmark_version": bench_cfg.benchmark_version,
        "primary_gate_tasks": list(bench_cfg.primary_gate_task_types()),
        "secondary_tasks": list(bench_cfg.secondary_eval_task_types()),
        "profile": profile,
        "train_context_length": train_context_length,
        "n_layers": config.get("model", {}).get("n_layers"),
        "dry_run": dry_run,
        "wall_sec": wall_sec,
        "peak_vram_mb": peak_mb,
        "training": train_info,
        "holdout_mid_train_samples": len(holdout_mid),
        "holdout_full_samples": len(holdout_full),
        "mid_train_samples_per_cell": per_cell,
        "staged_training_audit": audit,
        "final_eval": final_summary.to_dict(),
        "recommendation": recommendation,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "suite_step_override": {
            "dense_pretrain_steps": recommendation.get("recommended_steps"),
            "sparse_finetune_steps": recommendation.get("recommended_steps"),
        },
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary(payload, json_path, latest_path)
    return payload


def _print_summary(payload: dict, json_path: Path, latest_path: Path) -> None:
    rec = payload["recommendation"]
    final = payload["final_eval"]
    train = payload["training"]
    print()
    print("=== Dense calibration summary ===")
    print(f"  T={payload['train_context_length']}  layers={payload['n_layers']}  wall={payload['wall_sec']:.1f}s")
    print(f"  trained_steps={train.get('trained_steps')}")
    gate_acc = final.get("primary_gate_accuracy", final.get("pure_niah_accuracy"))
    gate_tasks = payload.get("primary_gate_tasks") or []
    secondary_tasks = payload.get("secondary_tasks") or []
    family = payload.get("benchmark_family", "nl")
    print(f"  benchmark: {family} v{payload.get('benchmark_version', '?')}")
    print(f"  primary gate accuracy: {_pct(gate_acc)}")
    print(f"  overall holdout accuracy: {final.get('overall_accuracy', 0.0) * 100:.2f}%")
    if final.get("secondary_total"):
        print(f"  secondary: {_pct(final.get('secondary_accuracy'))}")
    print(f"  best mid-train gate: step={rec.get('best_steps')} acc={_pct(rec.get('best_accuracy'))}")
    print(f"  recommended suite steps: {rec.get('recommended_steps')}")
    print(f"  verdict: {rec.get('verdict')} — {rec.get('reason')}")
    print()
    print("  Primary gate tasks (final):")
    for task in gate_tasks:
        acc = (final.get("by_task_type") or {}).get(task)
        print(f"    {task}: {_pct(acc)}")
    if secondary_tasks:
        print("  Secondary tasks (monitoring only):")
        for task in secondary_tasks:
            acc = (final.get("by_task_type") or {}).get(task)
            print(f"    {task}: {_pct(acc)}")
    print()
    print(f"  Wrote: {json_path}")
    print(f"  Latest: {latest_path}")
    print()
    steps = rec.get("recommended_steps")
    if steps:
        print("  Apply to suite:")
        print(
            f"    python run_experiment_7_suite.py --profile full "
            f"--calibration-steps-json {latest_path}"
        )


def _pct(v) -> str:
    if v is None:
        return "n/a"
    return f"{float(v) * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train dense_flash only on NIAH to calibrate suite step budget (standalone from variant suite).",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "fast"),
        default="fast",
        help="Suite profile to mirror: fast=2 layers (default), full=6 layers (Goals breakthrough)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Training context length (default: dense_calibration.train_context_length or 8192)",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Training step cap")
    parser.add_argument(
        "--validate-every",
        type=int,
        default=None,
        help="Holdout eval interval in steps (default from config, typically 1000)",
    )
    parser.add_argument(
        "--validate-every-min",
        type=int,
        default=None,
        help="Minimum allowed validate_every (default 500 for calibration)",
    )
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run (40 steps) to verify pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON reports (default: experiments/Experiment_7/dense_calibration)",
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save dense checkpoint after calibration (not used by suite)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic L0–L4 pointer/address benchmark (apply_synthetic_profile)",
    )
    args = parser.parse_args()

    log_dir = args.output_dir or (get_experiments_root() / "Experiment_7" / "dense_calibration")
    log_dir.mkdir(parents=True, exist_ok=True)
    global logger
    logger = setup_logging(log_dir, name="calibrate_dense_niah")
    _preflight()

    raw = load_config(ROOT / "configs" / "experiment_7.yaml")
    cal_defaults = raw.get("dense_calibration", {})
    train_t = args.context_length or int(cal_defaults.get("train_context_length", 8192))

    run_calibration(
        profile=args.profile,
        train_context_length=train_t,
        max_steps=args.max_steps,
        validate_every=args.validate_every,
        validate_every_min=args.validate_every_min,
        log_every=args.log_every,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        save_checkpoint=args.save_checkpoint,
        synthetic=args.synthetic,
    )


if __name__ == "__main__":
    main()
