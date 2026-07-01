#!/usr/bin/env python3
"""
Minimal-bet dense vs linear suite — three pre-registered experiments.

  RUN 1  niah_diagnostic_t2048           — 6L classic NIAH, restore_best (diagnostic)
  RUN 2  pointer_1decoy_first_wins_t2048 — same-key first-wins + 1 decoy scatter
  RUN 3  mqar_n4_q4_t2048                — MQAR N=4 Q=4, all-query supervision

Usage:
  python scripts/verify_dense_gap_minimal_bet.py
  python run_dense_gap_minimal_bet_suite.py --dry-run --variant all --experiment all
  python run_dense_gap_minimal_bet_suite.py --variant all --experiment all
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

CONFIG_DIR = ROOT / "configs" / "dense_gap_minimal_bet"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "dense_gap_minimal_bet"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_dense_gap_minimal_bet.py"

EXPERIMENTS: dict[str, Path] = {
    "niah_diagnostic_t2048": CONFIG_DIR / "niah_diagnostic_t2048.yaml",
    "pointer_1decoy_first_wins_t2048": CONFIG_DIR / "pointer_1decoy_first_wins_t2048.yaml",
    "mqar_n4_q4_t2048": CONFIG_DIR / "mqar_n4_q4_t2048.yaml",
}

RUN_ORDER = list(EXPERIMENTS)
SUPPORTED_VARIANTS = ("dense_flash", "linear")

RUN1_DENSE_MIN = 0.90
GAP_DENSE_MIN = 0.60
GAP_MIN_PP = 10.0


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== dense_gap_minimal_bet preflight ===")
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


def _official_accuracy(summary: dict | None) -> float | None:
    if not summary or summary.get("status") != "ok":
        return None
    ev = (summary.get("result") or {}).get("eval_official") or (summary.get("result") or {}).get(
        "eval"
    ) or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    return float(acc) if acc is not None else None


def _write_success_gate(experiment: str, dense_acc: float | None, linear_acc: float | None) -> dict:
    gap_pp = None
    if dense_acc is not None and linear_acc is not None:
        gap_pp = (dense_acc - linear_acc) * 100.0

    if experiment == "niah_diagnostic_t2048":
        criteria = {"dense_min": RUN1_DENSE_MIN, "purpose": "diagnostic_prerequisite"}
        success = dense_acc is not None and dense_acc >= RUN1_DENSE_MIN
    else:
        criteria = {
            "dense_min": GAP_DENSE_MIN,
            "gap_min_pp": GAP_MIN_PP,
        }
        success = (
            dense_acc is not None
            and linear_acc is not None
            and dense_acc >= GAP_DENSE_MIN
            and gap_pp is not None
            and gap_pp >= GAP_MIN_PP
        )

    gate = {
        "experiment": experiment,
        "dense_accuracy": dense_acc,
        "linear_accuracy": linear_acc,
        "gap_pp": gap_pp,
        "overall_success": success,
        "criteria": criteria,
    }
    out = OUTPUT_ROOT / experiment / "success_gate.json"
    out.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    return gate


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

    bench = _resolve_synthetic_bench_cfg(config, train_t)
    steps = int(config.get("transformer", {}).get("sparse_finetune_steps") or 0)

    print("=== Experiment plan ===")
    print(f"  experiment={experiment}")
    print(f"  task={bench.task_types[0]}  T={train_t}  scatter={bench.scatter_multi_needles}")
    print(f"  layers={n_layers}  steps={steps}  variant={variant}")
    print(f"  restore_best={cal.get('restore_best_checkpoint')}")
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
    log = logging.getLogger(f"minimal_bet.{experiment}.{variant}")
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
        restored = (payload.get("train_info") or {}).get("restored_best_checkpoint")
        acc_str = f"{acc_f * 100:.2f}%" if acc_f is not None else "n/a"
        print(f"OK {variant}: eval={acc_str} restored_best={restored}")
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
        "kind": "dense_gap_minimal_bet",
        "experiment": experiment,
        "variant": variant,
        "dry_run": dry_run,
        "train_context_length": train_t,
        "task_type": bench.task_types[0],
        "scatter_multi_needles": bench.scatter_multi_needles,
        "training_steps": steps,
        "restore_best_checkpoint": cal.get("restore_best_checkpoint"),
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


def _finalize_experiment(experiment: str) -> dict:
    dense_acc = _official_accuracy(
        json.loads((OUTPUT_ROOT / experiment / "dense_flash" / "latest.json").read_text())
        if (OUTPUT_ROOT / experiment / "dense_flash" / "latest.json").exists()
        else None
    )
    linear_acc = _official_accuracy(
        json.loads((OUTPUT_ROOT / experiment / "linear" / "latest.json").read_text())
        if (OUTPUT_ROOT / experiment / "linear" / "latest.json").exists()
        else None
    )
    gate = _write_success_gate(experiment, dense_acc, linear_acc)
    d_s = f"{dense_acc * 100:.2f}%" if dense_acc is not None else "n/a"
    l_s = f"{linear_acc * 100:.2f}%" if linear_acc is not None else "n/a"
    gap_s = f"{gate['gap_pp']:+.1f} pp" if gate["gap_pp"] is not None else "n/a"
    ok = "PASS" if gate["overall_success"] else "FAIL"
    print(f"  {experiment}: dense={d_s} linear={l_s} gap={gap_s} -> {ok}")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal-bet dense vs linear suite (3 runs)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--experiment", default="all", help="all or comma-separated name(s)")
    parser.add_argument("--variant", default="all", help="dense_flash | linear | all")
    args = parser.parse_args()

    if args.experiment == "all":
        names = RUN_ORDER
    else:
        names = [p.strip() for p in args.experiment.split(",") if p.strip()]
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {unknown}")

    variants = (
        list(SUPPORTED_VARIANTS)
        if args.variant == "all"
        else [args.variant]
        if args.variant in SUPPORTED_VARIANTS
        else (_ for _ in ()).throw(SystemExit(f"Unknown variant {args.variant!r}"))
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    suite_log = OUTPUT_ROOT / ("full_run_suite_dry.log" if args.dry_run else "full_run_suite.log")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    suite_fh = logging.FileHandler(suite_log, encoding="utf-8")
    suite_fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(suite_fh)

    preflight(variants, args.dry_run)
    total_errors = 0
    gates: list[dict] = []

    for name in names:
        print(f"\n########## EXPERIMENT: {name} ##########")
        for variant in variants:
            print(f"\n########## {name} / {variant} ##########")
            total_errors += run_experiment(
                name,
                config_path=EXPERIMENTS[name],
                dry_run=args.dry_run,
                skip_verify=args.skip_verify,
                variant=variant,
            )
        if len(variants) == 2 and not args.dry_run:
            gates.append(_finalize_experiment(name))

    logging.getLogger().removeHandler(suite_fh)
    suite_fh.close()

    print(f"\n=== Minimal-bet suite finished: {len(names)} experiments, {total_errors} errors ===")
    print(f"  suite log: {suite_log}")
    if gates:
        print("\n=== Success gates ===")
        for gate in gates:
            print(
                f"  {gate['experiment']}: success={gate['overall_success']} "
                f"(dense={gate.get('dense_accuracy')}, linear={gate.get('linear_accuracy')}, "
                f"gap_pp={gate.get('gap_pp')})"
            )

    suite_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "dense_gap_minimal_bet_suite",
        "dry_run": args.dry_run,
        "experiments": names,
        "variants": variants,
        "errors": total_errors,
        "gates": gates,
    }
    (OUTPUT_ROOT / "suite_latest.json").write_text(
        json.dumps(suite_summary, indent=2, default=str),
        encoding="utf-8",
    )
    sys.exit(1 if total_errors else 0)


if __name__ == "__main__":
    main()
