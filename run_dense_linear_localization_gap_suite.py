#!/usr/bin/env python3
"""
Pointer-scatter localization gap — dense vs linear (final-checkpoint eval).

Designed to show dense > linear on a task the 6L ~5M model can actually learn,
while staying on the path to longer contexts (T=1024 → 2048 → 4096).

Experiments:
  pointer_scatter_t1024_0decoy   — fast calibration
  pointer_scatter_t2048_0decoy   — PRIMARY (run this for the main result)
  pointer_scatter_t4096_0decoy   — optional scale-up (only if 2048 passes gates)

Pre-registered success (pointer_scatter_t2048_0decoy):
  - dense official accuracy >= 60%
  - dense minus linear >= 10 percentage points
  - dense wins at >= 3 of 5 depth buckets on holdout

Usage:
  python scripts/verify_dense_linear_localization_gap.py
  python run_dense_linear_localization_gap_suite.py --dry-run --variant all --experiment primary
  python run_dense_linear_localization_gap_suite.py --variant all --experiment pointer_scatter_t2048_0decoy
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import LongContextBenchmarkConfig
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    run_attention_baseline,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

CONFIG_DIR = ROOT / "configs" / "dense_linear_localization_gap"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "dense_linear_localization_gap"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_dense_linear_localization_gap.py"

EXPERIMENTS: dict[str, Path] = {
    "pointer_scatter_t1024_0decoy": CONFIG_DIR / "pointer_scatter_t1024_0decoy.yaml",
    "pointer_scatter_t2048_0decoy": CONFIG_DIR / "pointer_scatter_t2048_0decoy.yaml",
    "pointer_scatter_t4096_0decoy": CONFIG_DIR / "pointer_scatter_t4096_0decoy.yaml",
}

PRIMARY = "pointer_scatter_t2048_0decoy"
SUPPORTED_VARIANTS = ("dense_flash", "linear")

SUCCESS_DENSE_MIN = 0.60
SUCCESS_GAP_MIN_PP = 0.10
SUCCESS_DEPTH_BUCKETS_MIN = 3


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== dense_linear_localization_gap preflight ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — training will be slow on CPU.")
    try:
        assert_production_backends_available(variants)
    except (RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print()
    return info


def run_verify(experiment: str) -> None:
    print(f"=== Dataset verification ({experiment}) ===")
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--experiment", experiment],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print()


def _resolve_bench(config: dict, train_t: int) -> LongContextBenchmarkConfig:
    return _resolve_synthetic_bench_cfg(config, train_t)


def _run_variant(
    variant: str,
    *,
    config: dict,
    train_t: int,
    device: torch.device,
    log: logging.Logger,
) -> dict:
    if variant == "dense_flash":
        return run_dense_flash_finetune(
            config, train_t=train_t, dense_ckpt=None, device=device, log=log
        )
    if variant == "linear":
        return run_attention_baseline(
            config, variant, train_t=train_t, dense_ckpt=None, device=device, log=log
        )
    raise ValueError(f"Unsupported variant {variant!r}")


def _eval_metrics(payload: dict) -> dict:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    by_depth = ev.get("by_needle_depth") or {}
    depth_wins = 0
    depth_total = 0
    if isinstance(by_depth, dict):
        for _d, a in by_depth.items():
            if a is None:
                continue
            depth_total += 1
    return {
        "accuracy": float(acc) if acc is not None else None,
        "by_needle_depth": by_depth,
        "eval_subset": ev.get("eval_subset"),
    }


def _score_success(dense_acc: float | None, linear_acc: float | None, by_depth: dict) -> dict:
    gap_pp = None
    if dense_acc is not None and linear_acc is not None:
        gap_pp = (dense_acc - linear_acc) * 100.0
    depth_dense_wins = 0
    depth_pairs = 0
    if isinstance(by_depth, dict) and by_depth:
        dense_depth = by_depth if isinstance(by_depth, dict) else {}
    else:
        dense_depth = {}
    # by_depth in summary is dense-only; compare using combined latest if available
    dense_met = dense_acc is not None and dense_acc >= SUCCESS_DENSE_MIN
    gap_met = gap_pp is not None and gap_pp >= SUCCESS_GAP_MIN_PP
    return {
        "dense_min_met": dense_met,
        "gap_min_met": gap_met,
        "gap_pp": gap_pp,
        "dense_accuracy": dense_acc,
        "linear_accuracy": linear_acc,
        "depth_dense_wins": depth_dense_wins,
        "depth_pairs": depth_pairs,
        "overall_success": bool(dense_met and gap_met),
        "criteria": {
            "dense_official_min": SUCCESS_DENSE_MIN,
            "gap_min_pp": SUCCESS_GAP_MIN_PP,
            "depth_buckets_dense_wins_min": SUCCESS_DEPTH_BUCKETS_MIN,
        },
    }


def run_experiment(
    experiment: str,
    *,
    config_path: Path,
    dry_run: bool,
    skip_verify: bool,
    variant: str,
) -> int:
    if not skip_verify:
        run_verify(experiment)

    arena_cfg = load_routing_arena_config(config_path)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = int(arena_cfg.get("n_layers", 6))
    output_root = OUTPUT_ROOT / experiment
    output_root.mkdir(parents=True, exist_ok=True)

    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    cal = config.setdefault("dense_calibration", {})
    if dry_run:
        cal["eval_use_full_holdout"] = False
    else:
        cal.setdefault("eval_use_full_holdout", True)
    cal.setdefault("restore_best_checkpoint", False)

    bench = _resolve_bench(config, train_t)
    steps = int(config.get("transformer", {}).get("sparse_finetune_steps") or 0)

    print("=== Experiment plan ===")
    print(f"  experiment={experiment}")
    print(f"  task=pointer_unique  T={train_t}  scatter=True  decoys=0")
    print(f"  layers={n_layers}  steps={steps}  variant={variant}")
    print(f"  train_seed={bench.seed}  holdout_seed={bench.holdout_seed}")
    print(f"  output={output_root}")
    if dry_run:
        print("  dry-run: mid holdout only")
    print()

    variant_dir = output_root / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
    log_path = variant_dir / log_name
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)
    print(f"  log: {log_path}")

    device = init_arena_runtime(config)
    log = logging.getLogger(f"localization_gap.{experiment}.{variant}")
    reset_peak_vram(device)
    errors = 0
    try:
        payload = _run_variant(
            variant, config=config, train_t=train_t, device=device, log=log
        )
        metrics = _eval_metrics(payload)
        acc = metrics["accuracy"]
        acc_str = f"{acc * 100:.2f}%" if acc is not None else "n/a"
        print(f"OK {variant}: eval={acc_str} subset={metrics.get('eval_subset')}")
        status = "ok"
    except Exception:
        err = traceback.format_exc()
        print(err)
        payload = {"status": "error", "traceback": err}
        metrics = {}
        status = "error"
        errors = 1
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "dense_linear_localization_gap",
        "experiment": experiment,
        "variant": variant,
        "dry_run": dry_run,
        "train_context_length": train_t,
        "task_type": "pointer_unique",
        "scatter_multi_needles": True,
        "synthetic_decoy_keys": 0,
        "training_steps": steps,
        "train_seed": bench.seed,
        "holdout_seed": bench.holdout_seed,
        "result": payload,
        "metrics": metrics,
        "status": status,
    }
    summary_path = variant_dir / f"summary_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (variant_dir / "latest.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    combined_path = output_root / "combined_latest.json"
    combined: dict = {}
    if combined_path.exists():
        combined = json.loads(combined_path.read_text(encoding="utf-8"))
    combined[variant] = summary
    combined_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    print(f"  wrote: {summary_path}\n")
    return errors


def _resolve_experiment_names(raw: str) -> list[str]:
    if raw == "all":
        return list(EXPERIMENTS)
    if raw == "primary":
        return [PRIMARY]
    if raw == "calibrate":
        return ["pointer_scatter_t1024_0decoy"]
    if raw == "scale":
        return ["pointer_scatter_t4096_0decoy"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def _resolve_variants(raw: str) -> list[str]:
    if raw == "all":
        return list(SUPPORTED_VARIANTS)
    if raw not in SUPPORTED_VARIANTS:
        raise SystemExit(f"Unknown variant {raw!r}")
    return [raw]


def _print_success_report(experiment: str) -> None:
    root = OUTPUT_ROOT / experiment
    dense_p = root / "dense_flash" / "latest.json"
    linear_p = root / "linear" / "latest.json"
    if not dense_p.exists() or not linear_p.exists():
        return
    d = json.loads(dense_p.read_text(encoding="utf-8"))
    l = json.loads(linear_p.read_text(encoding="utf-8"))
    if d.get("status") != "ok" or l.get("status") != "ok":
        print(f"  {experiment}: incomplete (errors)")
        return

    d_ev = (d.get("result") or {}).get("eval_official") or (d.get("result") or {}).get("eval") or {}
    l_ev = (l.get("result") or {}).get("eval_official") or (l.get("result") or {}).get("eval") or {}
    d_acc = d_ev.get("primary_gate_accuracy", d_ev.get("overall_accuracy"))
    l_acc = l_ev.get("primary_gate_accuracy", l_ev.get("overall_accuracy"))
    d_depth = d_ev.get("by_needle_depth") or {}
    l_depth = l_ev.get("by_needle_depth") or {}

    gap_pp = (float(d_acc) - float(l_acc)) * 100 if d_acc is not None and l_acc is not None else None
    depth_wins = 0
    for key in d_depth:
        if key in l_depth:
            if float(d_depth[key]) > float(l_depth[key]):
                depth_wins += 1

    dense_met = d_acc is not None and float(d_acc) >= SUCCESS_DENSE_MIN
    gap_met = gap_pp is not None and gap_pp >= SUCCESS_GAP_MIN_PP
    depth_met = depth_wins >= SUCCESS_DEPTH_BUCKETS_MIN
    overall = dense_met and gap_met and depth_met

    print(f"\n=== Success gate: {experiment} ===")
    print(f"  dense={float(d_acc)*100:.1f}%  linear={float(l_acc)*100:.1f}%  gap={gap_pp:+.1f} pp")
    print(f"  depth buckets dense>linear: {depth_wins}/{len(d_depth)} (need >={SUCCESS_DEPTH_BUCKETS_MIN})")
    print(f"  dense>={SUCCESS_DENSE_MIN*100:.0f}%: {'PASS' if dense_met else 'FAIL'}")
    print(f"  gap>={SUCCESS_GAP_MIN_PP:.0f}pp: {'PASS' if gap_met else 'FAIL'}")
    print(f"  depth wins: {'PASS' if depth_met else 'FAIL'}")
    print(f"  OVERALL: {'SUCCESS' if overall else 'NOT YET'}")

    gate_path = root / "success_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "dense_accuracy": d_acc,
                "linear_accuracy": l_acc,
                "gap_pp": gap_pp,
                "depth_dense_wins": depth_wins,
                "overall_success": overall,
                "criteria": {
                    "dense_min": SUCCESS_DENSE_MIN,
                    "gap_min_pp": SUCCESS_GAP_MIN_PP,
                    "depth_buckets_min": SUCCESS_DEPTH_BUCKETS_MIN,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pointer-scatter dense vs linear gap suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--experiment",
        default="primary",
        help="primary | calibrate | scale | all | comma-separated name(s)",
    )
    parser.add_argument("--variant", default="all", help="dense_flash | linear | all")
    args = parser.parse_args()

    names = _resolve_experiment_names(args.experiment)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {unknown}")

    variants = _resolve_variants(args.variant)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    preflight(variants, args.dry_run)

    total_errors = 0
    for name in names:
        for variant in variants:
            print(f"\n########## {name} / {variant} ##########")
            total_errors += run_experiment(
                name,
                config_path=EXPERIMENTS[name],
                dry_run=args.dry_run,
                skip_verify=args.skip_verify,
                variant=variant,
            )

    for name in names:
        if len(variants) >= 2 or args.variant == "all":
            _print_success_report(name)

    print(f"\n=== Suite finished: {len(names)} x {len(variants)}, {total_errors} errors ===")
    sys.exit(min(total_errors, 1))


if __name__ == "__main__":
    main()
