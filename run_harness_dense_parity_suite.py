#!/usr/bin/env python3
"""
Harness reproduction suite — feasibility ladder vs arena parity.

Phase A: Run unchanged feasibility_ladder_4L_20k (confirm ~95% dense still works).
Phase B: Run arena dense-only with feasibility_parity fixes (must match ~95%).

Usage:
  python scripts/diff_feasibility_vs_arena_config.py
  python run_harness_dense_parity_suite.py --phase feasibility --dry-run
  python run_harness_dense_parity_suite.py --phase arena --dry-run
  python run_harness_dense_parity_suite.py --phase all
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

from routing_attention.benchmarks.long_context.feasibility_ladder import (
    run_feasibility_ladder,
)
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

FEASIBILITY_CFG = ROOT / "configs" / "feasibility_ladder_4L_20k.yaml"
ARENA_PARITY_CFG = ROOT / "configs" / "harness_dense_parity" / "niah_4L_t2048.yaml"
DIFF_SCRIPT = ROOT / "scripts" / "diff_feasibility_vs_arena_config.py"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "harness_dense_parity"
DENSE_MIN = 0.90


def preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== harness_dense_parity preflight ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available.")
    try:
        assert_production_backends_available(["dense_flash"])
    except (RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print()
    return info


def run_config_diff() -> None:
    print("=== Config diff (feasibility vs old arena) ===")
    proc = subprocess.run(
        [sys.executable, str(DIFF_SCRIPT)],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print()


def run_feasibility_phase(*, dry_run: bool) -> dict:
    out_dir = OUTPUT_ROOT / ("feasibility_repro_dry" if dry_run else "feasibility_repro")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n########## PHASE A: feasibility ladder (unchanged) ##########")
    print(f"  config={FEASIBILITY_CFG}")
    print(f"  output={out_dir}")
    payload = run_feasibility_ladder(
        dry_run=dry_run,
        levels=[2],
        config_path=FEASIBILITY_CFG,
        output_dir=out_dir,
        stop_on_failure=False,
    )
    level = (payload.get("levels") or [{}])[0]
    eval_d = level.get("final_eval") or {}
    acc = float(
        eval_d.get(
            "primary_gate_accuracy",
            eval_d.get("overall_accuracy", 0.0),
        )
    )
    print(f"  feasibility dense official acc: {acc * 100:.2f}%")
    return {
        "phase": "feasibility",
        "dry_run": dry_run,
        "accuracy": acc,
        "passed": acc >= DENSE_MIN,
        "payload": payload,
    }


def run_arena_phase(*, dry_run: bool) -> dict:
    out_dir = OUTPUT_ROOT / ("arena_parity_dry" if dry_run else "arena_parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    arena_cfg = load_routing_arena_config(ARENA_PARITY_CFG)
    n_layers = int(arena_cfg.get("n_layers", 4))
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    train_t = int(arena_cfg["train_context_length"])
    bench = _resolve_synthetic_bench_cfg(config, train_t)

    print(f"\n########## PHASE B: arena dense parity ##########")
    print(f"  config={ARENA_PARITY_CFG}")
    print(f"  feasibility_parity={arena_cfg.get('feasibility_parity')}")
    print(f"  model.vocab_size={config.get('model', {}).get('vocab_size')}")
    print(f"  steps={config.get('transformer', {}).get('max_steps')}")
    print(f"  output={out_dir}")

    variant_dir = out_dir / "dense_flash"
    variant_dir.mkdir(parents=True, exist_ok=True)
    log_path = variant_dir / ("run_dry_dense_flash.log" if dry_run else "run_dense_flash.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    device = init_arena_runtime(config)
    log = logging.getLogger("arena_parity.dense_flash")
    reset_peak_vram(device)
    errors = 0
    try:
        payload = run_dense_flash_finetune(
            config,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
            save_checkpoint_path=variant_dir / "dense_flash.pt",
        )
        ev = payload.get("eval_official") or payload.get("eval") or {}
        acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
        acc_f = float(acc) if acc is not None else None
        acc_s = f"{acc_f * 100:.2f}%" if acc_f is not None else "n/a"
        print(f"  arena dense official acc: {acc_s}")
        status = "ok"
    except Exception:
        print(traceback.format_exc())
        payload = {"status": "error"}
        acc_f = None
        status = "error"
        errors = 1
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "harness_arena_parity",
        "dry_run": dry_run,
        "train_context_length": train_t,
        "n_layers": n_layers,
        "model_vocab_size": config.get("model", {}).get("vocab_size"),
        "bench_vocab_size": bench.vocab_size,
        "result": payload,
        "official_accuracy": acc_f,
        "status": status,
    }
    (variant_dir / "latest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return {
        "phase": "arena",
        "dry_run": dry_run,
        "accuracy": acc_f,
        "passed": acc_f is not None and acc_f >= DENSE_MIN,
        "errors": errors,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness dense parity reproduction")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase", choices=("feasibility", "arena", "all"), default="all")
    parser.add_argument("--skip-diff", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not args.skip_diff:
        run_config_diff()
    preflight(args.dry_run)

    results: list[dict] = []
    if args.phase in ("feasibility", "all"):
        results.append(run_feasibility_phase(dry_run=args.dry_run))
    if args.phase in ("arena", "all"):
        results.append(run_arena_phase(dry_run=args.dry_run))

    gate = {
        "dense_min": DENSE_MIN,
        "dry_run": args.dry_run,
        "phases": results,
        "feasibility_passed": next(
            (r["passed"] for r in results if r["phase"] == "feasibility"), None
        ),
        "arena_passed": next((r["passed"] for r in results if r["phase"] == "arena"), None),
        "harness_parity_achieved": all(r.get("passed") for r in results if not args.dry_run),
    }
    (OUTPUT_ROOT / "success_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

    print("\n=== Harness reproduction summary ===")
    for r in results:
        acc = r.get("accuracy")
        acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
        print(f"  {r['phase']}: {acc_s}  passed={r.get('passed')}")
    print(f"  gate: {OUTPUT_ROOT / 'success_gate.json'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
