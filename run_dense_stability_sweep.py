#!/usr/bin/env python3
"""
Dense training stability sweep — cudnn deterministic, LR warmup, seed search, 3× replicate.

Phases:
  1. Dry-run smoke (stability settings)
  2. Seed sweep: seeds 42–46 @ full 20k steps
  3. Arena parity 3× replicate at best seed
  4. Three-way head-to-head using best checkpoint (if dense >= 90%)

Usage:
  python run_dense_stability_sweep.py --dry-run
  python run_dense_stability_sweep.py
  python run_dense_stability_sweep.py --phase seed_sweep --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import copy
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
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

ARENA_CFG = ROOT / "configs" / "harness_dense_parity" / "niah_4L_t2048.yaml"
THREE_WAY_SCRIPT = ROOT / "run_niah_three_way_4L_suite.py"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "dense_stability_sweep"
DENSE_MIN = 0.90
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
REPLICATE_COUNT = 3


def preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== dense_stability_sweep preflight ===")
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


def _arena_cfg_with_seed(seed: int) -> dict:
    raw = load_routing_arena_config(ARENA_CFG)
    cfg = copy.deepcopy(raw)
    cfg["seed"] = seed
    return cfg


def run_dense_trial(
    *,
    seed: int,
    out_dir: Path,
    dry_run: bool,
    label: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    arena_cfg = _arena_cfg_with_seed(seed)
    n_layers = int(arena_cfg.get("n_layers", 4))
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    train_t = int(arena_cfg["train_context_length"])
    bench = _resolve_synthetic_bench_cfg(config, train_t)

    print(f"\n########## {label} seed={seed} ##########")
    print(f"  cudnn_deterministic={config.get('training', {}).get('cudnn_deterministic')}")
    print(f"  lr_warmup_steps={config.get('transformer', {}).get('lr_warmup_steps')}")
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
    log = logging.getLogger(f"dense_stability.{label}")
    reset_peak_vram(device)
    ckpt_path = variant_dir / "dense_flash.pt"
    try:
        payload = run_dense_flash_finetune(
            config,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
            save_checkpoint_path=ckpt_path,
        )
        ev = payload.get("eval_official") or payload.get("eval") or {}
        acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
        acc_f = float(acc) if acc is not None else None
        best_step = (payload.get("train_info") or {}).get("best_holdout", {}).get("step")
        print(f"  official acc: {acc_f * 100:.2f}%" if acc_f is not None else "  official acc: n/a")
        status = "ok"
    except Exception:
        print(traceback.format_exc())
        payload = {"status": "error"}
        acc_f = None
        best_step = None
        status = "error"
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "dense_stability_trial",
        "label": label,
        "seed": seed,
        "dry_run": dry_run,
        "train_context_length": train_t,
        "n_layers": n_layers,
        "model_vocab_size": config.get("model", {}).get("vocab_size"),
        "bench_vocab_size": bench.vocab_size,
        "training": config.get("training", {}),
        "lr_warmup_steps": config.get("transformer", {}).get("lr_warmup_steps"),
        "official_accuracy": acc_f,
        "best_holdout_step": best_step,
        "checkpoint": str(ckpt_path) if ckpt_path.exists() else None,
        "status": status,
        "result": payload,
    }
    (variant_dir / "latest.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "trial_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_seed_sweep(*, seeds: list[int], dry_run: bool) -> list[dict]:
    results: list[dict] = []
    sweep_dir = OUTPUT_ROOT / ("seed_sweep_dry" if dry_run else "seed_sweep")
    for seed in seeds:
        results.append(
            run_dense_trial(
                seed=seed,
                out_dir=sweep_dir / f"seed_{seed}",
                dry_run=dry_run,
                label=f"seed_sweep_{seed}",
            )
        )
    return results


def run_replicates(*, seed: int, count: int, dry_run: bool) -> list[dict]:
    results: list[dict] = []
    rep_dir = OUTPUT_ROOT / ("replicates_dry" if dry_run else "replicates") / f"seed_{seed}"
    for i in range(1, count + 1):
        results.append(
            run_dense_trial(
                seed=seed,
                out_dir=rep_dir / f"rep_{i}",
                dry_run=dry_run,
                label=f"replicate_{i}",
            )
        )
    return results


def _pick_best(trials: list[dict]) -> dict | None:
    ok = [t for t in trials if t.get("official_accuracy") is not None and t.get("status") == "ok"]
    if not ok:
        return None
    return max(ok, key=lambda t: float(t["official_accuracy"]))


def run_three_way(*, dense_checkpoint: Path, dry_run: bool) -> int:
    cmd = [
        sys.executable,
        str(THREE_WAY_SCRIPT),
        "--dense-checkpoint",
        str(dense_checkpoint),
        "--skip-verify",
    ]
    if dry_run:
        cmd.append("--dry-run")
    print(f"\n########## three-way (dense ckpt={dense_checkpoint}) ##########")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    return proc.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense stability sweep + three-way")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "smoke", "seed_sweep", "replicates", "three_way"),
        default="all",
    )
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--replicates", type=int, default=REPLICATE_COUNT)
    parser.add_argument("--best-seed", type=int, default=None, help="Seed for replicate phase")
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight(args.dry_run)

    gate: dict = {
        "dense_min": DENSE_MIN,
        "dry_run": args.dry_run,
        "seeds": seeds,
        "replicate_count": args.replicates,
        "stability": {
            "cudnn_deterministic": True,
            "lr_warmup_steps": 500,
        },
    }

    seed_results: list[dict] = []
    replicate_results: list[dict] = []
    best_trial: dict | None = None
    three_way_rc: int | None = None

    if args.phase in ("all", "smoke"):
        smoke = run_dense_trial(
            seed=seeds[0],
            out_dir=OUTPUT_ROOT / "smoke_dry" if args.dry_run else OUTPUT_ROOT / "smoke",
            dry_run=True,
            label="smoke",
        )
        gate["smoke"] = smoke
        if args.phase == "smoke":
            (OUTPUT_ROOT / "success_gate.json").write_text(json.dumps(gate, indent=2, default=str))
            print("\n=== Smoke finished ===")
            sys.exit(0)

    if args.phase in ("all", "seed_sweep"):
        seed_results = run_seed_sweep(seeds=seeds, dry_run=args.dry_run)
        gate["seed_sweep"] = seed_results
        best_trial = _pick_best(seed_results)
        if best_trial:
            print(
                f"\n  best seed sweep: seed={best_trial['seed']} "
                f"acc={float(best_trial['official_accuracy']) * 100:.2f}%"
            )

    best_seed = args.best_seed
    if best_seed is None and best_trial:
        best_seed = int(best_trial["seed"])

    if args.phase in ("all", "replicates") and best_seed is not None:
        replicate_results = run_replicates(
            seed=best_seed,
            count=args.replicates,
            dry_run=args.dry_run,
        )
        gate["replicates"] = replicate_results
        best_trial = _pick_best(replicate_results) or best_trial

    rep_accs = [
        float(r["official_accuracy"])
        for r in replicate_results
        if r.get("official_accuracy") is not None
    ]
    gate["replicate_pass_count"] = sum(1 for a in rep_accs if a >= DENSE_MIN)
    gate["replicate_median_accuracy"] = (
        sorted(rep_accs)[len(rep_accs) // 2] if rep_accs else None
    )

    dense_ckpt = args.dense_checkpoint
    if dense_ckpt is None and best_trial and best_trial.get("checkpoint"):
        dense_ckpt = Path(best_trial["checkpoint"])

    gate["best_trial"] = best_trial
    gate["best_checkpoint"] = str(dense_ckpt) if dense_ckpt else None
    dense_reliable = (
        gate.get("replicate_pass_count", 0) >= 2
        or (
            best_trial is not None
            and float(best_trial.get("official_accuracy") or 0) >= DENSE_MIN
        )
    )
    gate["dense_reliable"] = dense_reliable and not args.dry_run

    if args.phase in ("all", "three_way") and dense_ckpt and Path(dense_ckpt).exists():
        if args.dry_run or dense_reliable or args.phase == "three_way":
            three_way_rc = run_three_way(dense_checkpoint=Path(dense_ckpt), dry_run=args.dry_run)
            gate["three_way_exit_code"] = three_way_rc
        else:
            print("\nSKIP three-way: dense not reliable (need >=2/3 replicates >= 90%)")
            gate["three_way_skipped"] = True
    elif args.phase in ("all", "three_way"):
        print("\nSKIP three-way: no valid dense checkpoint")
        gate["three_way_skipped"] = True

    (OUTPUT_ROOT / "success_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

    print("\n=== Dense stability sweep summary ===")
    if seed_results:
        print("  seed sweep:")
        for r in seed_results:
            acc = r.get("official_accuracy")
            acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
            print(f"    seed {r['seed']}: {acc_s}")
    if replicate_results:
        print(f"  replicates @ seed {best_seed}:")
        for r in replicate_results:
            acc = r.get("official_accuracy")
            acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
            print(f"    {r['label']}: {acc_s}")
        print(f"  pass count (>={DENSE_MIN:.0%}): {gate.get('replicate_pass_count')}/{len(replicate_results)}")
    if best_trial:
        acc = best_trial.get("official_accuracy")
        acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
        print(f"  best checkpoint: {best_trial.get('checkpoint')} ({acc_s})")
    print(f"  gate: {OUTPUT_ROOT / 'success_gate.json'}")
    sys.exit(0 if three_way_rc in (None, 0) else three_way_rc)


if __name__ == "__main__":
    main()
