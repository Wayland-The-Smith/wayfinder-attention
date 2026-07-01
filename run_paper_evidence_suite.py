#!/usr/bin/env python3
"""
Paper evidence suite — reproducibility, systems benchmarks, routing decoy tune, length scaling.

Phases (run in order):
  1. probe       — train-step feasibility at long T (16384 first)
  2. seed_repro  — 0-decoy head-to-head @ T=2048, seeds 43/45/46
  3. systems     — forward latency + VRAM @ T=2048..16384, all variants
  4. routing_tune — close routing-vs-dense gap @ decoys=1 (from curriculum dense ckpt)
  5. length      — train+eval 0-decoy, longest→shortest (feasible T only)

Usage:
  python run_paper_evidence_suite.py --phase probe
  python run_paper_evidence_suite.py --phase all --dry-run
  python run_paper_evidence_suite.py --phase all
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
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paper_evidence_common import (
    DEFAULT_DENSE_CKPT,
    HEADTOHEAD_VARIANTS,
    LENGTH_LEVELS_L2S,
    PAPER_OUTPUT,
    ROUTING_LR,
    ROUTING_TOP_K,
    SEED_REPRO_SEEDS,
    SYSTEMS_LENGTHS,
    build_cell_config,
    official_accuracy,
    train_feasible,
)
from routing_attention.benchmarks.long_context.evaluation import LongContextEvaluator
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    _resolve_synthetic_bench_cfg,
    init_arena_runtime,
    run_attention_baseline,
    run_dense_flash_eval_from_checkpoint,
    run_dense_flash_finetune,
    run_key_vector_k32,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, peak_vram_mb, reset_peak_vram
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.cuda import configure_cuda_training

PROBE_SCRIPT = ROOT / "scripts" / "probe_length_train_feasibility.py"
PROBE_JSON = PAPER_OUTPUT / "feasibility_probe" / "train_step_probe.json"
DECOY_DENSE_CKPT = (
    PAPER_OUTPUT.parent / "decoy_curriculum" / "decoys_1" / "checkpoints" / "dense_flash.pt"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def preflight() -> None:
    configure_cuda_training({"training": {"cudnn_deterministic": True, "cudnn_benchmark": False}})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    print("=== paper_evidence preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    assert_production_backends_available(list(HEADTOHEAD_VARIANTS))
    print()


def run_probe_phase() -> dict:
    out_dir = PAPER_OUTPUT / "feasibility_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--lengths",
        "16384,8192,4096,2048",
        "--layers",
        "4,2,1",
        "--variants",
        "dense_flash,linear,key_vector_k32",
        "--output",
        str(PROBE_JSON),
    ]
    print("Running train-step feasibility probe (16384 first)...")
    proc = subprocess.run(cmd, cwd=ROOT, env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)})
    if proc.returncode != 0:
        raise RuntimeError(f"probe failed exit={proc.returncode}")
    return json.loads(PROBE_JSON.read_text(encoding="utf-8"))


def _load_probe_recommendations() -> dict[int, int | None]:
    if not PROBE_JSON.exists():
        return {t: 4 for t in LENGTH_LEVELS_L2S}
    data = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    out: dict[int, int | None] = {}
    for rec in data.get("recommendations", []):
        out[int(rec["train_t"])] = rec.get("n_layers")
    return out


def _feasible_lengths(probe: dict | None) -> list[int]:
    if probe is None:
        probe = json.loads(PROBE_JSON.read_text(encoding="utf-8")) if PROBE_JSON.exists() else {}
    feasible: list[int] = []
    for rec in probe.get("recommendations", []):
        t = int(rec["train_t"])
        n = rec.get("n_layers")
        p = rec.get("probe") or {}
        if n is not None and p.get("feasible"):
            feasible.append(t)
    # Always include 2048
    if 2048 not in feasible:
        feasible.append(2048)
    return sorted(set(feasible), reverse=True)


def run_variant_cell(
    *,
    variant: str,
    config: dict,
    train_t: int,
    device: torch.device,
    dense_ckpt: Path | None,
    save_dense: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    log = logging.getLogger(f"paper.{variant}.T{train_t}")
    reset_peak_vram(device)
    try:
        if variant == "dense_flash":
            if dense_ckpt and dense_ckpt.exists() and save_dense is None:
                payload = run_dense_flash_eval_from_checkpoint(
                    config, train_t=train_t, dense_ckpt=dense_ckpt, device=device, log=log
                )
            else:
                payload = run_dense_flash_finetune(
                    config,
                    train_t=train_t,
                    dense_ckpt=dense_ckpt,
                    device=device,
                    log=log,
                    save_checkpoint_path=save_dense,
                )
        elif variant == "key_vector_k32":
            if not dense_ckpt or not dense_ckpt.exists():
                raise FileNotFoundError(f"routing needs dense ckpt: {dense_ckpt}")
            payload = run_key_vector_k32(
                config,
                train_t=train_t,
                dense_ckpt=dense_ckpt,
                device=device,
                log=log,
                top_k=ROUTING_TOP_K,
            )
        else:
            payload = run_attention_baseline(
                config,
                variant,
                train_t=train_t,
                dense_ckpt=dense_ckpt,
                device=device,
                log=log,
            )
        return {"status": "ok", "accuracy": official_accuracy(payload), "result": payload}
    except Exception:
        print(traceback.format_exc())
        return {"status": "error", "accuracy": None, "traceback": traceback.format_exc()}
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_seed_repro_phase(*, dry_run: bool) -> dict:
    out_root = PAPER_OUTPUT / ("seed_repro_dry" if dry_run else "seed_repro")
    out_root.mkdir(parents=True, exist_ok=True)
    print("\n########## seed repro (0 decoy @ T=2048) ##########")

    all_results: list[dict] = []
    for seed in SEED_REPRO_SEEDS:
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        config = build_cell_config(
            train_t=2048,
            decoys=0,
            dry_run=dry_run,
            n_layers=4,
            seed=seed,
            tag=f"seed_repro_s{seed}",
            train_steps=20000 if not dry_run else None,
        )
        device = init_arena_runtime(config)
        dense_ckpt = DEFAULT_DENSE_CKPT if seed == 45 and DEFAULT_DENSE_CKPT.exists() else None
        cell: dict[str, Any] = {"seed": seed, "variants": {}}

        for variant in HEADTOHEAD_VARIANTS:
            print(f"\n  seed={seed} {variant}")
            init = None
            save = None
            if variant == "dense_flash":
                if dense_ckpt:
                    init = dense_ckpt
                else:
                    init = None
                    save = seed_dir / "dense_flash.pt"
            elif variant == "key_vector_k32":
                dpath = seed_dir / "dense_flash.pt"
                if dpath.exists():
                    init = dpath
                elif dense_ckpt:
                    init = dense_ckpt
                else:
                    cell["variants"][variant] = {"status": "skipped", "reason": "no_dense_ckpt"}
                    continue
            res = run_variant_cell(
                variant=variant,
                config=config,
                train_t=2048,
                device=device,
                dense_ckpt=init,
                save_dense=save,
                dry_run=dry_run,
            )
            if variant == "dense_flash" and res.get("status") == "ok" and save is None and dense_ckpt:
                import shutil

                seed_dir.mkdir(parents=True, exist_ok=True)
                dst = seed_dir / "dense_flash.pt"
                shutil.copy2(dense_ckpt, dst)
            cell["variants"][variant] = res
            acc = res.get("accuracy")
            print(f"    acc={acc * 100:.2f}%" if acc is not None else "    acc=n/a")

        (seed_dir / "cell_summary.json").write_text(json.dumps(cell, indent=2, default=str), encoding="utf-8")
        all_results.append(cell)

    summary = {
        "seeds": SEED_REPRO_SEEDS,
        "dry_run": dry_run,
        "cells": all_results,
        "timestamp": _now(),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_systems_phase(*, dry_run: bool) -> dict:
    out_root = PAPER_OUTPUT / ("systems_dry" if dry_run else "systems")
    out_root.mkdir(parents=True, exist_ok=True)
    print("\n########## systems latency + VRAM ##########")

    config = build_cell_config(
        train_t=2048,
        decoys=0,
        dry_run=True,
        n_layers=4,
        seed=45,
        tag="systems_bench",
    )
    device = init_arena_runtime(config)
    bench_cfg = _resolve_synthetic_bench_cfg(config, 2048)
    holdout = []  # latency only

    rows: list[dict] = []
    for train_t in SYSTEMS_LENGTHS:
        print(f"\n  T={train_t}")
        for variant in HEADTOHEAD_VARIANTS:
            reset_peak_vram(device)
            try:
                from experiments.experiment_7 import _build_variant_model

                model, _ = _build_variant_model(
                    config,
                    variant,
                    device,
                    train_t,
                    dense_checkpoint=DEFAULT_DENSE_CKPT if DEFAULT_DENSE_CKPT.exists() else None,
                )
                model.eval()
                evaluator = LongContextEvaluator(bench_cfg, holdout_samples=holdout)
                lat = evaluator.benchmark_forward_latency(
                    model,
                    device=device,
                    context_length=train_t,
                    warmup=1 if dry_run else 2,
                    runs=2 if dry_run else 5,
                )
                vram = peak_vram_mb(device)
                row = {
                    "variant": variant,
                    "context_length": train_t,
                    "latency_ms": lat.get("latency_ms"),
                    "tokens_per_sec": lat.get("tokens_per_sec"),
                    "peak_vram_mb": vram,
                    "error": lat.get("error"),
                }
                print(f"    {variant}: {lat.get('latency_ms')} ms  vram={vram}MB")
            except Exception as exc:
                row = {
                    "variant": variant,
                    "context_length": train_t,
                    "error": str(exc)[:300],
                }
                print(f"    {variant}: ERROR {exc}")
            rows.append(row)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summary = {"dry_run": dry_run, "rows": rows, "timestamp": _now()}
    (out_root / "systems_benchmark.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_routing_tune_phase(*, dry_run: bool) -> dict:
    out_root = PAPER_OUTPUT / ("routing_decoy_tune_dry" if dry_run else "routing_decoy_tune")
    out_root.mkdir(parents=True, exist_ok=True)
    print("\n########## routing decoy gap tune (d=1) ##########")

    dense_init = DECOY_DENSE_CKPT if DECOY_DENSE_CKPT.exists() else DEFAULT_DENSE_CKPT
    trials: list[dict] = []
    steps_list = [10000, 20000] if not dry_run else [80]
    lr_list = [3e-4, 1e-4] if not dry_run else [3e-4]

    config_base = build_cell_config(
        train_t=2048,
        decoys=1,
        dry_run=dry_run,
        n_layers=4,
        seed=45,
        tag="routing_decoy_tune",
    )

    for steps in steps_list:
        for lr in lr_list:
            label = f"k{ROUTING_TOP_K}_s{steps}_lr{lr:g}"
            trial_dir = out_root / label
            trial_dir.mkdir(parents=True, exist_ok=True)
            config = copy.deepcopy(config_base)
            config["key_vector"]["sparse_finetune_steps"] = steps
            config["key_vector"]["sparse_finetune_lr"] = lr
            config["transformer"]["sparse_finetune_steps"] = steps
            device = init_arena_runtime(config)
            print(f"\n  trial {label}")
            res = run_variant_cell(
                variant="key_vector_k32",
                config=config,
                train_t=2048,
                device=device,
                dense_ckpt=dense_init,
                save_dense=None,
                dry_run=dry_run,
            )
            dense_eval = None
            if dense_init.exists():
                dense_cfg = build_cell_config(
                    train_t=2048, decoys=1, dry_run=False, n_layers=4, seed=45, tag="dense_ref"
                )
                init_arena_runtime(dense_cfg)
                de = run_dense_flash_eval_from_checkpoint(
                    dense_cfg,
                    train_t=2048,
                    dense_ckpt=dense_init,
                    device=device,
                    log=logging.getLogger("dense_ref"),
                )
                dense_eval = official_accuracy(de)
            routing_acc = res.get("accuracy")
            gap_pp = (routing_acc - dense_eval) * 100 if routing_acc and dense_eval else None
            trial = {
                "label": label,
                "steps": steps,
                "lr": lr,
                "routing_accuracy": routing_acc,
                "dense_reference_accuracy": dense_eval,
                "routing_minus_dense_pp": gap_pp,
                "result": res,
            }
            trials.append(trial)
            (trial_dir / "summary.json").write_text(json.dumps(trial, indent=2, default=str), encoding="utf-8")
            print(f"    routing={routing_acc} dense_ref={dense_eval} gap_pp={gap_pp}")

    best = max(
        (t for t in trials if t.get("routing_accuracy") is not None),
        key=lambda t: float(t["routing_accuracy"]),
        default=None,
    )
    summary = {"trials": trials, "best": best, "timestamp": _now(), "dry_run": dry_run}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def run_length_phase(*, dry_run: bool, probe: dict | None) -> dict:
    out_root = PAPER_OUTPUT / ("length_scaling_dry" if dry_run else "length_scaling")
    out_root.mkdir(parents=True, exist_ok=True)
    print("\n########## length scaling (longest first) ##########")

    if probe is None and PROBE_JSON.exists():
        probe = json.loads(PROBE_JSON.read_text(encoding="utf-8"))

    feasible = _feasible_lengths(probe)
    rec_layers = _load_probe_recommendations()
    print(f"  feasible lengths: {feasible}")

    cells: list[dict] = []
    for train_t in LENGTH_LEVELS_L2S:
        n_layers = rec_layers.get(train_t, 4)
        if train_t not in feasible and train_t != 2048:
            print(f"\n  SKIP T={train_t} (probe: not feasible)")
            cells.append({"train_t": train_t, "skipped": True, "reason": "probe_infeasible"})
            continue

        if n_layers is None:
            print(f"\n  SKIP T={train_t} (no viable layer count)")
            cells.append({"train_t": train_t, "skipped": True, "reason": "no_layers"})
            continue

        cell_dir = out_root / f"T{train_t}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        steps = 10000 if not dry_run else 80
        config = build_cell_config(
            train_t=train_t,
            decoys=0,
            dry_run=dry_run,
            n_layers=n_layers,
            seed=45,
            tag=f"length_T{train_t}",
            train_steps=steps,
        )
        device = init_arena_runtime(config)
        print(f"\n  T={train_t} n_layers={n_layers} steps={steps}")

        cell: dict[str, Any] = {"train_t": train_t, "n_layers": n_layers, "variants": {}}
        dense_ckpt_path = cell_dir / "dense_flash.pt"

        for variant in HEADTOHEAD_VARIANTS:
            init = None
            save = None
            if variant == "dense_flash":
                init = None
                save = dense_ckpt_path
            elif variant == "key_vector_k32":
                init = dense_ckpt_path if dense_ckpt_path.exists() else None
                if not init:
                    cell["variants"][variant] = {"status": "skipped"}
                    continue
            res = run_variant_cell(
                variant=variant,
                config=config,
                train_t=train_t,
                device=device,
                dense_ckpt=init,
                save_dense=save,
                dry_run=dry_run,
            )
            cell["variants"][variant] = res
            acc = res.get("accuracy")
            print(f"    {variant}: {acc * 100:.2f}%" if acc else f"    {variant}: n/a")

        (cell_dir / "cell_summary.json").write_text(json.dumps(cell, indent=2, default=str), encoding="utf-8")
        cells.append(cell)

    summary = {"cells": cells, "feasible_lengths": feasible, "dry_run": dry_run, "timestamp": _now()}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def write_manifest(gate: dict) -> Path:
    PAPER_OUTPUT.mkdir(parents=True, exist_ok=True)
    gate["updated_at"] = _now()
    path = PAPER_OUTPUT / "paper_results_manifest.json"
    path.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper evidence suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-probe", action="store_true", help="Reuse existing feasibility_probe JSON")
    parser.add_argument(
        "--phase",
        choices=("all", "probe", "seed_repro", "systems", "routing_tune", "length"),
        default="all",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PAPER_OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight()

    gate: dict[str, Any] = {"dry_run": args.dry_run, "phases": {}}
    probe_data: dict | None = None

    if args.phase in ("all", "probe") and not args.skip_probe:
        probe_data = run_probe_phase()
        gate["phases"]["probe"] = probe_data.get("recommendations", [])
    elif args.skip_probe and PROBE_JSON.exists():
        probe_data = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
        gate["phases"]["probe"] = probe_data.get("recommendations", [])

    if args.phase in ("all", "length") and probe_data is None and PROBE_JSON.exists():
        probe_data = json.loads(PROBE_JSON.read_text(encoding="utf-8"))

    if args.phase in ("all", "seed_repro"):
        gate["phases"]["seed_repro"] = run_seed_repro_phase(dry_run=args.dry_run)

    if args.phase in ("all", "systems"):
        gate["phases"]["systems"] = run_systems_phase(dry_run=args.dry_run)

    if args.phase in ("all", "routing_tune"):
        gate["phases"]["routing_tune"] = run_routing_tune_phase(dry_run=args.dry_run)

    if args.phase in ("all", "length"):
        gate["phases"]["length_scaling"] = run_length_phase(dry_run=args.dry_run, probe=probe_data)

    manifest = write_manifest(gate)
    print(f"\n=== Paper evidence manifest: {manifest} ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
