#!/usr/bin/env python3
"""
Dense vs linear head-to-head gap experiments (final-checkpoint eval protocol).

Experiments (6L, 40k steps, restore_best_checkpoint=false, official 300 holdout):
  conflict_scatter_t1024_rows5       — last-wins, scatter @ T=1024, 5 rows
  conflict_first_scatter_t2048_rows4 — first-wins, scatter @ T=2048, 4 rows
  conflict_middle_bunched_t512_rows6 — middle-wins, bunched @ T=512, 6 rows

Usage:
  python scripts/verify_dense_gap_headtohead.py
  python run_dense_gap_headtohead_suite.py --dry-run --variant all --experiment all
  python run_dense_gap_headtohead_suite.py --dry-run --variant dense_flash --experiment conflict_scatter_t1024_rows5
  python run_dense_gap_headtohead_suite.py --variant linear --experiment conflict_middle_bunched_t512_rows6
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

CONFIG_DIR = ROOT / "configs" / "dense_gap_headtohead"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "dense_gap_headtohead"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_dense_gap_headtohead.py"

EXPERIMENTS: dict[str, Path] = {
    "conflict_scatter_t1024_rows5": CONFIG_DIR / "conflict_scatter_t1024_rows5.yaml",
    "conflict_first_scatter_t2048_rows4": CONFIG_DIR / "conflict_first_scatter_t2048_rows4.yaml",
    "conflict_middle_bunched_t512_rows6": CONFIG_DIR / "conflict_middle_bunched_t512_rows6.yaml",
}

SUPPORTED_VARIANTS = ("dense_flash", "linear")
DEFAULT_VARIANT = "dense_flash"


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== dense_gap_headtohead preflight ===")
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
    family = str(config.get("long_context_benchmark", {}).get("benchmark_family", "synthetic"))
    if family == "synthetic":
        return _resolve_synthetic_bench_cfg(config, train_t)
    return LongContextBenchmarkConfig.from_dict(config["long_context_benchmark"]).normalized()


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
            config,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
        )
    if variant == "linear":
        return run_attention_baseline(
            config,
            variant,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
        )
    raise ValueError(f"Unsupported variant {variant!r}")


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
    transformer_cfg = config.get("transformer", {})
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("dense_pretrain_steps")
        or 0
    )

    print("=== Experiment plan ===")
    print(f"  experiment={experiment}")
    print(f"  task={bench.task_types[0]}  T={train_t}  scatter={bench.scatter_multi_needles}")
    print(f"  conflict_rows={bench.synthetic_conflict_rows}  family={bench.benchmark_family}")
    print(f"  layers={n_layers}  steps={steps}  variant={variant}")
    print(f"  train_seed={bench.seed}  holdout_seed={bench.holdout_seed}")
    print(f"  restore_best={cal.get('restore_best_checkpoint')}  full_holdout={cal.get('eval_use_full_holdout')}")
    print(f"  output={output_root}")
    if dry_run:
        print("  dry-run: mid holdout only (no official 300 eval)")
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
    log = logging.getLogger(f"dense_gap_headtohead.{experiment}.{variant}")
    reset_peak_vram(device)
    errors = 0
    try:
        payload = _run_variant(
            variant,
            config=config,
            train_t=train_t,
            device=device,
            log=log,
        )
        ev = payload.get("eval_official") or payload.get("eval", {})
        acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
        acc_f = float(acc) if acc is not None else None
        subset = ev.get("eval_subset", "unknown")
        restored = (payload.get("train_info") or {}).get("restored_best_checkpoint")
        acc_str = f"{acc_f * 100:.2f}%" if acc_f is not None else "n/a"
        print(f"OK {variant}: eval={acc_str} subset={subset} restored_best={restored}")
        status = "ok"
    except Exception:
        err = traceback.format_exc()
        print(err)
        payload = {"status": "error", "traceback": err}
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
        "kind": "dense_gap_headtohead",
        "experiment": experiment,
        "variant": variant,
        "dry_run": dry_run,
        "train_context_length": train_t,
        "task_type": bench.task_types[0],
        "benchmark_family": bench.benchmark_family,
        "scatter_multi_needles": bench.scatter_multi_needles,
        "synthetic_conflict_rows": bench.synthetic_conflict_rows,
        "training_steps": steps,
        "train_seed": bench.seed,
        "holdout_seed": bench.holdout_seed,
        "result": payload,
        "status": status,
    }
    summary_path = variant_dir / f"summary_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (variant_dir / "latest.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
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
    return [p.strip() for p in raw.split(",") if p.strip()]


def _resolve_variants(raw: str) -> list[str]:
    if raw == "all":
        return list(SUPPORTED_VARIANTS)
    if raw not in SUPPORTED_VARIANTS:
        raise SystemExit(f"Unknown variant {raw!r}; use dense_flash, linear, or all")
    return [raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense vs linear gap head-to-head experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--experiment", default="all", help="all or comma-separated name(s)")
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help="dense_flash | linear | all",
    )
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

    if len(names) > 1 or len(variants) > 1:
        print(f"\n=== Suite finished: {len(names)} experiments x {len(variants)} variants, {total_errors} errors ===")
        for name in names:
            for variant in variants:
                latest = OUTPUT_ROOT / name / variant / "latest.json"
                if not latest.exists():
                    print(f"  {name}/{variant}: (missing)")
                    continue
                summary = json.loads(latest.read_text(encoding="utf-8"))
                if summary.get("status") != "ok":
                    print(f"  {name}/{variant}: ERROR")
                    continue
                ev = (summary.get("result") or {}).get("eval_official") or (
                    summary.get("result") or {}
                ).get("eval") or {}
                acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
                acc_s = f"{float(acc) * 100:.2f}%" if acc is not None else "n/a"
                print(f"  {name}/{variant}: {acc_s}")
            dense_latest = OUTPUT_ROOT / name / "dense_flash" / "latest.json"
            linear_latest = OUTPUT_ROOT / name / "linear" / "latest.json"
            if dense_latest.exists() and linear_latest.exists():
                d = json.loads(dense_latest.read_text(encoding="utf-8"))
                l = json.loads(linear_latest.read_text(encoding="utf-8"))
                if d.get("status") == "ok" and l.get("status") == "ok":
                    dev = (d.get("result") or {}).get("eval_official") or (d.get("result") or {}).get("eval") or {}
                    lev = (l.get("result") or {}).get("eval_official") or (l.get("result") or {}).get("eval") or {}
                    da = dev.get("primary_gate_accuracy", dev.get("overall_accuracy"))
                    la = lev.get("primary_gate_accuracy", lev.get("overall_accuracy"))
                    if da is not None and la is not None:
                        print(f"  {name} dense-linear gap: {(float(da) - float(la)) * 100:.2f} pp")

    sys.exit(min(total_errors, 1))


if __name__ == "__main__":
    main()
