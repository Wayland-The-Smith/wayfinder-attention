#!/usr/bin/env python3
"""
Learned-address breakthrough suite — Phases 1–3 orchestrator.

Phase 1: sweep + curriculum (no recall gate), linear baseline, reuse dense ckpt
Phase 2: seed repro (43/45/46), depth-stratified metrics, hard cell (1 decoy)
Phase 3: unified results table + claim manifest

Usage:
  python run_learned_address_breakthrough_suite.py --dry-run
  python run_learned_address_breakthrough_suite.py --phase all
  python run_learned_address_breakthrough_suite.py --phase feasibility
  python run_learned_address_breakthrough_suite.py --phase phase1 --dense-checkpoint path/to/dense.pt
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

from learned_address_proof_common import (
    BREAKTHROUGH_CLAIM,
    BREAKTHROUGH_OUTPUT,
    CONFIG_PATH,
    HARD_CELL_CONFIG_PATH,
    PHASE_C_VARIANTS,
    PROOF_CELL_DENSE_CKPT,
    SEED_REPRO_SEEDS,
    SYSTEMS_LENGTHS,
    SYSTEMS_VARIANTS,
    depth_stratified,
    official_accuracy,
    post_phase_c_recall,
    recall_at_k,
    resolve_dense_checkpoint,
    write_json,
)
from run_learned_address_proof_cell_suite import (
    _load_arena,
    preflight,
    run_curriculum,
    run_phase_b,
    run_phase_c_variant,
    run_proof_cell,
    run_sweep,
    run_systems_benchmark,
    run_verify,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    init_arena_runtime,
    load_routing_arena_config,
    run_dense_flash_eval_from_checkpoint,
    run_dense_flash_finetune,
)
from routing_attention.utils.cuda import configure_cuda_training

PROBE_SCRIPT = ROOT / "scripts" / "probe_length_train_feasibility.py"
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_breakthrough_results.py"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_learned_address_proof_cell.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag(dry_run: bool) -> str:
    return "dry" if dry_run else "full"


def run_feasibility(*, dry_run: bool) -> dict[str, Any]:
    out_dir = BREAKTHROUGH_OUTPUT / "feasibility"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("probe_dry.json" if dry_run else "probe_full.json")
    lengths = "8192,4096" if dry_run else "8192,4096,2048"
    variants = "dense_flash,linear,key_vector_k32"
    cmd = [
        sys.executable,
        str(PROBE_SCRIPT),
        "--lengths",
        lengths,
        "--layers",
        "4",
        "--variants",
        variants,
        "--output",
        str(out_path),
    ]
    print(f"\n########## Feasibility probe ##########")
    print(" ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        print(f"WARNING: feasibility probe exit={proc.returncode} — continuing")
        return {"warning": "probe_failed", "exit_code": proc.returncode}
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    write_json(out_dir / "latest.json", payload)
    for rec in payload.get("recommendations", []):
        print(
            f"  T={rec['train_t']} n_layers={rec.get('n_layers')} "
            f"feasible={rec.get('probe', {}).get('feasible')} "
            f"step_ms={rec.get('probe', {}).get('step_ms_mean')}"
        )
    return payload


def run_linear_extension(*, dry_run: bool, dense_ckpt: Path) -> dict[str, Any]:
    """Add linear baseline to proof cell using existing dense teacher (Phase C linear only)."""
    tag = _tag(dry_run)
    out_dir = BREAKTHROUGH_OUTPUT / f"phase1_linear_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    arena_cfg, config, _ = _load_arena(dry_run)
    train_t = int(arena_cfg["train_context_length"])
    device = init_arena_runtime(config)
    log = logging.getLogger("breakthrough.linear")

    print("\n########## Phase 1: linear baseline @ T=2048 ##########")
    payload = run_phase_c_variant(
        "linear",
        config=config,
        train_t=train_t,
        dense_ckpt=dense_ckpt,
        address_idx=None,
        device=device,
        log=log,
        out_dir=out_dir,
        dry_run=dry_run,
    )
    summary = {
        "phase": "phase1_linear",
        "dry_run": dry_run,
        "timestamp": _now(),
        "linear_accuracy": official_accuracy(payload),
        "by_needle_depth": depth_stratified(payload),
        "result": payload,
    }
    write_json(out_dir / "summary.json", summary)
    write_json(BREAKTHROUGH_OUTPUT / f"phase1_linear_{tag}_latest.json", summary)
    return summary


def run_full_protocol_cell(
    *,
    name: str,
    config_path: Path,
    dry_run: bool,
    dense_ckpt: Path | None,
    seed: int,
    train_t: int | None = None,
    out_subdir: str,
) -> dict[str, Any]:
    """Dense (optional reuse) → B → all Phase C variants."""
    tag = _tag(dry_run)
    out_dir = BREAKTHROUGH_OUTPUT / out_subdir / (f"seed_{seed}" if "seed" in out_subdir else name) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    arena_cfg, config, proof = _load_arena(dry_run, train_t=train_t, config_path=config_path)
    if seed != int(arena_cfg.get("seed", seed)):
        arena_cfg = copy.deepcopy(arena_cfg)
        arena_cfg["seed"] = seed
        from routing_attention.benchmarks.long_context.routing_arena import build_arena_experiment_config

        config = build_arena_experiment_config(
            arena_cfg, dry_run=dry_run, n_layers=int(arena_cfg["n_layers"])
        )
    train_t = int(arena_cfg["train_context_length"])
    recall_k = int(proof.get("recall_k", 128))
    device = init_arena_runtime(config)
    log = logging.getLogger(f"breakthrough.{name}.seed{seed}")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dense_path = ckpt_dir / ("dense_flash_dry.pt" if dry_run else "dense_flash.pt")

    cell: dict[str, Any] = {
        "name": name,
        "seed": seed,
        "train_t": train_t,
        "dry_run": dry_run,
        "timestamp": _now(),
    }

    if dense_ckpt and dense_ckpt.exists() and seed == 45 and train_t == 2048 and name != "seed_repro":
        print(f"  dense eval-only from {dense_ckpt}")
        phase_a = run_dense_flash_eval_from_checkpoint(
            config, train_t=train_t, dense_ckpt=dense_ckpt, device=device, log=log
        )
        dense_ckpt = Path(dense_ckpt)
    elif dense_ckpt and dense_ckpt.exists() and train_t != 2048:
        print(f"  dense warm-start from {dense_ckpt}")
        phase_a = run_dense_flash_finetune(
            config,
            train_t=train_t,
            dense_ckpt=dense_ckpt,
            device=device,
            log=log,
            save_checkpoint_path=dense_path,
        )
        dense_ckpt = Path(phase_a.get("saved_dense_checkpoint") or dense_path)
    else:
        phase_a = run_dense_flash_finetune(
            config,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
            save_checkpoint_path=dense_path,
        )
        dense_ckpt = Path(phase_a.get("saved_dense_checkpoint") or dense_path)

    cell["phase_a"] = phase_a
    cell["dense_accuracy"] = official_accuracy(phase_a)
    cell["dense_checkpoint"] = str(dense_ckpt)

    phase_b = run_phase_b(
        config=config,
        arena_cfg=arena_cfg,
        dense_ckpt=dense_ckpt,
        train_t=train_t,
        device=device,
        out_dir=out_dir,
        dry_run=dry_run,
        force_refresh=True,
    )
    cell["phase_b"] = phase_b
    cell[f"recall@{recall_k}_after_b"] = phase_b.get(f"recall@{recall_k}")
    address_idx = Path(phase_b["address_index_checkpoint"])

    phase_c: dict[str, Any] = {}
    for variant in PHASE_C_VARIANTS:
        try:
            phase_c[variant] = run_phase_c_variant(
                variant,
                config=config,
                train_t=train_t,
                dense_ckpt=dense_ckpt,
                address_idx=address_idx,
                device=device,
                log=log,
                out_dir=out_dir,
                dry_run=dry_run,
            )
        except Exception:
            err = traceback.format_exc()
            print(err)
            phase_c[variant] = {"status": "error", "traceback": err}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cell["phase_c"] = phase_c
    cell["metrics"] = {
        v: {
            "accuracy": official_accuracy(p),
            "by_needle_depth": depth_stratified(p),
            "recall_after_c": post_phase_c_recall(p, recall_k) if v == "learned_address_k32" else None,
        }
        for v, p in phase_c.items()
        if isinstance(p, dict)
    }
    write_json(out_dir / "cell.json", cell)
    return cell


def run_phase1(*, dry_run: bool, dense_ckpt: Path | None) -> dict[str, Any]:
    tag = _tag(dry_run)
    out: dict[str, Any] = {"phase": "phase1", "dry_run": dry_run, "timestamp": _now(), "steps": {}}
    dense = dense_ckpt or resolve_dense_checkpoint()

    if dense is None or not dense.exists():
        print("No dense checkpoint — running proof cell Phase A first")
        out["steps"]["proof_cell"] = run_proof_cell(
            dry_run=dry_run, dense_checkpoint=None, force_index=True
        )
        dense = Path(out["steps"]["proof_cell"].get("dense_checkpoint", ""))
    else:
        print(f"Reusing dense checkpoint: {dense}")
        out["steps"]["dense_reuse"] = str(dense)

    if dense and Path(dense).exists():
        out["steps"]["linear"] = run_linear_extension(dry_run=dry_run, dense_ckpt=Path(dense))

    out["steps"]["sweep"] = run_sweep(dry_run=dry_run, dense_checkpoint=Path(dense) if dense else None)
    out["steps"]["curriculum"] = run_curriculum(
        dry_run=dry_run, dense_checkpoint=Path(dense) if dense else None
    )

    if dense and Path(dense).exists():
        arena_cfg, config, _ = _load_arena(dry_run)
        train_t = int(arena_cfg["train_context_length"])
        device = init_arena_runtime(config)
        index_dir = BREAKTHROUGH_OUTPUT.parent / "learned_address_proof_cell" / f"proof_cell_{tag}"
        addr = index_dir / "index_checkpoints" / f"T{train_t}_address_index.pt"
        if not addr.exists():
            addr = (
                ROOT
                / "experiments"
                / "Experiment_7"
                / "learned_address_proof_cell"
                / "proof_cell_full"
                / "index_checkpoints"
                / f"T{train_t}_address_index.pt"
            )
        out["steps"]["systems"] = run_systems_benchmark(
            config=config,
            train_t=train_t,
            dense_ckpt=Path(dense),
            address_idx=addr if addr.exists() else None,
            device=device,
            out_dir=BREAKTHROUGH_OUTPUT / f"systems_{tag}",
            dry_run=dry_run,
        )

    write_json(BREAKTHROUGH_OUTPUT / f"phase1_{tag}.json", out)
    return out


def run_phase2(*, dry_run: bool, dense_ckpt: Path | None) -> dict[str, Any]:
    tag = _tag(dry_run)
    out: dict[str, Any] = {"phase": "phase2", "dry_run": dry_run, "timestamp": _now()}

    seed_cells: dict[str, Any] = {}
    canonical_dense = dense_ckpt or resolve_dense_checkpoint()
    for seed in SEED_REPRO_SEEDS:
        print(f"\n########## Phase 2 seed repro seed={seed} ##########")
        reuse = canonical_dense if seed == 45 and canonical_dense and canonical_dense.exists() else None
        try:
            seed_cells[str(seed)] = run_full_protocol_cell(
                name="seed_repro",
                config_path=CONFIG_PATH,
                dry_run=dry_run,
                dense_ckpt=reuse,
                seed=seed,
                out_subdir="phase2_seed_repro",
            )
        except Exception:
            err = traceback.format_exc()
            print(err)
            seed_cells[str(seed)] = {"error": err}
    out["seed_repro"] = seed_cells

    print("\n########## Phase 2 hard cell (1 decoy) ##########")
    try:
        out["hard_cell"] = run_full_protocol_cell(
            name="hard_cell_d1",
            config_path=HARD_CELL_CONFIG_PATH,
            dry_run=dry_run,
            dense_ckpt=None,
            seed=45,
            out_subdir="phase2_hard_cell",
        )
    except Exception:
        out["hard_cell"] = {"error": traceback.format_exc()}

    depth_rows: list[dict] = []
    for label, cell in [
        ("proof_cell", BREAKTHROUGH_OUTPUT.parent / "learned_address_proof_cell" / "proof_cell_full" / "manifest.json"),
        ("phase1_linear", BREAKTHROUGH_OUTPUT / f"phase1_linear_{tag}_latest.json"),
    ]:
        path = Path(cell)
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if label == "proof_cell":
            for variant, payload in (data.get("phase_c") or {}).items():
                depth_rows.append(
                    {
                        "source": label,
                        "variant": variant,
                        "accuracy": official_accuracy(payload),
                        "by_needle_depth": depth_stratified(payload),
                    }
                )
        else:
            depth_rows.append(
                {
                    "source": label,
                    "variant": "linear",
                    "accuracy": data.get("linear_accuracy"),
                    "by_needle_depth": data.get("by_needle_depth", {}),
                }
            )
    for seed, cell in seed_cells.items():
        if not isinstance(cell, dict):
            continue
        for variant, metrics in (cell.get("metrics") or {}).items():
            depth_rows.append(
                {
                    "source": f"seed_repro_{seed}",
                    "variant": variant,
                    **metrics,
                }
            )
    out["depth_stratified"] = depth_rows
    write_json(BREAKTHROUGH_OUTPUT / f"phase2_{tag}.json", out)
    write_json(BREAKTHROUGH_OUTPUT / f"depth_stratified_{tag}.json", {"rows": depth_rows})
    return out


def run_phase3(*, dry_run: bool) -> dict[str, Any]:
    tag = _tag(dry_run)
    print("\n########## Phase 3: aggregate results ##########")
    cmd = [
        sys.executable,
        str(AGGREGATE_SCRIPT),
        "--input",
        str(BREAKTHROUGH_OUTPUT),
        "--output",
        str(BREAKTHROUGH_OUTPUT / f"results_table_{tag}.json"),
    ]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        raise RuntimeError(f"aggregate failed exit={proc.returncode}")
    table_path = BREAKTHROUGH_OUTPUT / f"results_table_{tag}.json"
    table = json.loads(table_path.read_text(encoding="utf-8"))
    manifest = {
        "phase": "phase3",
        "dry_run": dry_run,
        "timestamp": _now(),
        "claim": BREAKTHROUGH_CLAIM,
        "results_table": str(table_path),
        "table": table,
    }
    write_json(BREAKTHROUGH_OUTPUT / f"breakthrough_manifest_{tag}.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned-address breakthrough suite (Phases 1–3)")
    parser.add_argument(
        "--phase",
        choices=("feasibility", "phase1", "phase2", "phase3", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--dense-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    configure_cuda_training({"training": {"cudnn_deterministic": True, "cudnn_benchmark": False}})
    BREAKTHROUGH_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = BREAKTHROUGH_OUTPUT / ("master_dry.log" if args.dry_run else "master_full.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().addHandler(fh)

    if not args.skip_verify and VERIFY_SCRIPT.exists():
        run_verify()
    preflight(args.dry_run)

    dense = args.dense_checkpoint or resolve_dense_checkpoint()
    if dense:
        print(f"Using dense checkpoint: {dense}")

    phases = (
        ["feasibility", "phase1", "phase2", "phase3"]
        if args.phase == "all"
        else [args.phase]
    )
    results: dict[str, Any] = {"dry_run": args.dry_run, "phases": {}, "timestamp": _now()}
    exit_code = 0

    for phase in phases:
        print(f"\n{'=' * 70}\nBREAKTHROUGH {phase.upper()}  dry_run={args.dry_run}\n{'=' * 70}")
        try:
            if phase == "feasibility":
                results["phases"][phase] = run_feasibility(dry_run=args.dry_run)
            elif phase == "phase1":
                results["phases"][phase] = run_phase1(dry_run=args.dry_run, dense_ckpt=dense)
            elif phase == "phase2":
                results["phases"][phase] = run_phase2(dry_run=args.dry_run, dense_ckpt=dense)
            else:
                results["phases"][phase] = run_phase3(dry_run=args.dry_run)
        except Exception:
            err = traceback.format_exc()
            print(err)
            results["phases"][phase] = {"error": err}
            exit_code = 1

    write_json(BREAKTHROUGH_OUTPUT / f"master_{_tag(args.dry_run)}.json", results)
    logging.getLogger().removeHandler(fh)
    fh.close()
    print(f"\nMaster log: {log_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
