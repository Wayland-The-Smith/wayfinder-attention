#!/usr/bin/env python3
"""
Progress-efficient breakthrough gate @ T=8192.

Finishes the minimum experiment set for the long-context claim:
  - Phase B + C learned-address (resume curriculum T=8192, reuse dense + attention cache)
  - Phase C key_vector + local_window64 baselines
  - Systems benchmark (2048–16384)
  - Results aggregation

Skips all completed T=2048/4096 work and mega-suite Phase 2.

Usage:
  python run_breakthrough_gate_suite.py --dry-run
  python run_breakthrough_gate_suite.py
  python run_breakthrough_gate_suite.py --skip-verify
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
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from learned_address_proof_common import (
    BREAKTHROUGH_CLAIM,
    BREAKTHROUGH_OUTPUT,
    OUTPUT_ROOT,
    depth_stratified,
    official_accuracy,
    post_phase_c_recall,
    write_json,
)
from run_learned_address_proof_cell_suite import (
    VERIFY_SCRIPT,
    _load_arena,
    preflight,
    run_phase_b,
    run_phase_c_variant,
    run_systems_benchmark,
    run_verify,
)
from routing_attention.benchmarks.long_context.routing_arena import init_arena_runtime

GATE_TRAIN_T = 8192
GATE_VARIANTS = ("key_vector_k32", "learned_address_k32", "local_window64")
EXTRAP_EVAL_T = 16384
AGGREGATE_SCRIPT = ROOT / "scripts" / "aggregate_breakthrough_results.py"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tag(dry_run: bool) -> str:
    return "dry" if dry_run else "full"


def _cell_dir(*, dry_run: bool) -> Path:
    sub = "curriculum_dry" if dry_run else "curriculum_full"
    return OUTPUT_ROOT / sub / f"T{GATE_TRAIN_T}"


def _dense_ckpt_path(*, dry_run: bool) -> Path:
    return _cell_dir(dry_run=dry_run) / "checkpoints" / "dense_flash.pt"


def _load_cell(*, dry_run: bool) -> dict[str, Any]:
    path = _cell_dir(dry_run=dry_run) / "cell.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _eval_extrapolation(
    *,
    config: dict,
    arena_cfg: dict,
    train_t: int,
    eval_t: int,
    dense_ckpt: Path,
    address_idx: Path | None,
    device: torch.device,
    dry_run: bool,
) -> dict[str, Any]:
    """Holdout accuracy at eval_t using weights trained at train_t (position extrapolation)."""
    from experiments.experiment_4 import _expand_position_embeddings
    from experiments.experiment_7 import _build_variant_model
    from routing_attention.benchmarks.long_context.routing_arena import (
        _arena_holdout_splits,
        _official_eval,
        _load_address_index_into_model,
    )

    rows: dict[str, Any] = {}
    bench_cfg, _, holdout_full, _ = _arena_holdout_splits(arena_cfg, eval_t)
    if dry_run:
        holdout_full = holdout_full[:8]

    for variant in ("dense_flash", "learned_address_k32", "key_vector_k32", "local_window64"):
        try:
            if variant == "learned_address_k32" and (address_idx is None or not address_idx.exists()):
                continue
            model, var_config = _build_variant_model(
                config,
                variant,
                device,
                train_t,
                dense_checkpoint=dense_ckpt,
            )
            _expand_position_embeddings(model, eval_t)
            if variant == "learned_address_k32" and address_idx is not None:
                _load_address_index_into_model(model, var_config, address_idx, device)
            model.eval()
            result = _official_eval(model, bench_cfg, holdout_full, device)
            acc = official_accuracy(result)
            rows[variant] = {
                "accuracy": acc,
                "by_needle_depth": result.get("by_needle_depth", {}),
                "eval_t": eval_t,
                "train_t": train_t,
            }
            print(f"  extrap T={eval_t} {variant}: {acc * 100:.2f}%" if acc is not None else f"  extrap {variant}: n/a")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as exc:
            rows[variant] = {"error": str(exc), "traceback": traceback.format_exc()}
            print(f"  extrap {variant}: ERROR {exc}")
    return rows


def _reset_phase_b_artifacts(cell_dir: Path, *, keep_layer0: bool = True) -> None:
    """Remove partial index artifacts so full Phase B retrains incomplete layers."""
    index_dir = cell_dir / "index_checkpoints"
    addr_pt = index_dir / f"T{GATE_TRAIN_T}_address_index.pt"
    if addr_pt.exists():
        addr_pt.unlink()
        print(f"  removed stale address index: {addr_pt}")
    addr_ckpt = index_dir / f"_address_cache_T{GATE_TRAIN_T}" / "checkpoints" / "addresses"
    if addr_ckpt.exists():
        for layer_dir in sorted(addr_ckpt.glob("layer_*")):
            if keep_layer0 and layer_dir.name == "layer_0":
                print(f"  keeping completed layer checkpoint: {layer_dir}")
                continue
            final_pt = layer_dir / "final.pt"
            if final_pt.exists():
                final_pt.unlink()
                print(f"  removed partial layer checkpoint: {final_pt}")


def run_gate_cell(*, dry_run: bool, force_phase_b: bool = False) -> dict[str, Any]:
    tag = _tag(dry_run)
    cell_dir = _cell_dir(dry_run=dry_run)
    cell_dir.mkdir(parents=True, exist_ok=True)
    arena_cfg, config, _ = _load_arena(dry_run, train_t=GATE_TRAIN_T)
    device = init_arena_runtime(config)
    log = logging.getLogger(f"gate.T{GATE_TRAIN_T}")
    recall_k = int(arena_cfg.get("proof_cell", {}).get("recall_k", 128))

    cell: dict[str, Any] = {"train_t": GATE_TRAIN_T, "dry_run": dry_run, "timestamp": _now()}
    dense_path = _dense_ckpt_path(dry_run=dry_run)

    print(f"\n{'=' * 70}\nBREAKTHROUGH GATE T={GATE_TRAIN_T}  dry_run={dry_run}\n{'=' * 70}")

    if force_phase_b and not dry_run:
        print("\n########## Reset partial Phase B artifacts ##########")
        _reset_phase_b_artifacts(cell_dir, keep_layer0=True)

    # Phase A — skip if dense checkpoint already exists (full run only).
    if not dry_run and dense_path.exists():
        print(f"\n########## Phase A: SKIP (dense exists) ##########\n  {dense_path}")
        prev = _load_cell(dry_run=False).get("phase_a", {})
        cell["phase_a"] = {
            "skipped": True,
            "saved_dense_checkpoint": str(dense_path),
            "dense_accuracy": prev.get("eval_official", {}).get("overall_accuracy")
            or _load_cell(dry_run=False).get("dense_accuracy"),
        }
        dense_ckpt = dense_path
        cell["dense_accuracy"] = cell["phase_a"].get("dense_accuracy")
    else:
        from routing_attention.benchmarks.long_context.routing_arena import run_dense_flash_finetune

        t4096 = OUTPUT_ROOT / "curriculum_full" / "T4096" / "checkpoints" / "dense_flash.pt"
        warm = t4096 if t4096.exists() and not dry_run else None
        print(f"\n########## Phase A: dense T={GATE_TRAIN_T} ##########")
        if warm:
            print(f"  warm-start from {warm}")
            phase_a = run_dense_flash_finetune(
                config,
                train_t=GATE_TRAIN_T,
                dense_ckpt=warm,
                device=device,
                log=log,
                save_checkpoint_path=dense_path,
            )
        else:
            phase_a = run_dense_flash_finetune(
                config,
                train_t=GATE_TRAIN_T,
                dense_ckpt=None,
                device=device,
                log=log,
                save_checkpoint_path=dense_path,
            )
        dense_ckpt = Path(phase_a.get("saved_dense_checkpoint") or dense_path)
        cell["phase_a"] = phase_a
        cell["dense_accuracy"] = official_accuracy(phase_a)

    # Phase B — reuse attention cache when present; skip completed layers via trainer.
    try:
        print(f"\n########## Phase B: address index T={GATE_TRAIN_T} ##########")
        phase_b = run_phase_b(
            config=config,
            arena_cfg=arena_cfg,
            dense_ckpt=dense_ckpt,
            train_t=GATE_TRAIN_T,
            device=device,
            out_dir=cell_dir,
            dry_run=dry_run,
            force_refresh=False,
        )
        address_idx = Path(phase_b["address_index_checkpoint"])
        cell["phase_b"] = phase_b
    except Exception:
        err = traceback.format_exc()
        cell["phase_b"] = {"error": err}
        cell["error"] = err
        print(err)
        write_json(cell_dir / "cell.json", cell)
        write_json(BREAKTHROUGH_OUTPUT / f"gate_{tag}.json", cell)
        raise

    # Phase C — learned-address + baselines.
    phase_c: dict[str, Any] = {}
    for variant in GATE_VARIANTS:
        try:
            addr = address_idx if variant == "learned_address_k32" else None
            phase_c[variant] = run_phase_c_variant(
                variant,
                config=config,
                train_t=GATE_TRAIN_T,
                dense_ckpt=dense_ckpt,
                address_idx=addr,
                device=device,
                log=log,
                out_dir=cell_dir,
                dry_run=dry_run,
            )
        except Exception:
            err = traceback.format_exc()
            phase_c[variant] = {"error": err, "traceback": err}
            print(err)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cell["phase_c"] = phase_c
    cell["learned_address"] = phase_c.get("learned_address_k32")
    cell["learned_address_accuracy"] = official_accuracy(phase_c.get("learned_address_k32", {}))

    cell["metrics"] = {
        v: {
            "accuracy": official_accuracy(p) if isinstance(p, dict) else None,
            "by_needle_depth": depth_stratified(p) if isinstance(p, dict) else {},
            "recall_after_c": post_phase_c_recall(p, recall_k)
            if v == "learned_address_k32" and isinstance(p, dict)
            else None,
        }
        for v, p in phase_c.items()
    }

    # Length extrapolation quality @ T=16384.
    print(f"\n########## Extrapolation eval train={GATE_TRAIN_T} eval={EXTRAP_EVAL_T} ##########")
    cell["extrapolation"] = _eval_extrapolation(
        config=config,
        arena_cfg=arena_cfg,
        train_t=GATE_TRAIN_T,
        eval_t=EXTRAP_EVAL_T,
        dense_ckpt=dense_ckpt,
        address_idx=address_idx,
        device=device,
        dry_run=dry_run,
    )

    # Pass/fail gates for the breakthrough claim.
    dense_acc = float(cell.get("dense_accuracy") or 0.0)
    la_acc = float(cell.get("learned_address_accuracy") or 0.0)
    kv_acc = float(official_accuracy(phase_c.get("key_vector_k32", {})) or 0.0)
    local_acc = float(official_accuracy(phase_c.get("local_window64", {})) or 0.0)
    cell["gates"] = {
        "dense_passed": dense_acc >= 0.90,
        "learned_address_near_dense": abs(la_acc - dense_acc) <= 0.03,
        "beats_key_vector": (la_acc - kv_acc) >= 0.10,
        "beats_local": (la_acc - local_acc) >= 0.50,
        "dense_accuracy": dense_acc,
        "learned_address_accuracy": la_acc,
        "key_vector_accuracy": kv_acc,
        "local_window64_accuracy": local_acc,
        "passed": (
            dense_acc >= 0.90
            and abs(la_acc - dense_acc) <= 0.03
            and (la_acc - kv_acc) >= 0.10
            and (la_acc - local_acc) >= 0.50
        ),
    }
    print(f"\n  GATE passed={cell['gates']['passed']}")
    print(f"    dense={dense_acc:.1%}  la={la_acc:.1%}  kv={kv_acc:.1%}  local={local_acc:.1%}")

    write_json(cell_dir / "cell.json", cell)
    gate_out = BREAKTHROUGH_OUTPUT / f"gate_t8192_{tag}"
    gate_out.mkdir(parents=True, exist_ok=True)
    write_json(gate_out / "gate_cell.json", cell)
    return cell


def run_gate_systems(*, dry_run: bool, cell: dict[str, Any]) -> dict[str, Any]:
    arena_cfg, config, _ = _load_arena(dry_run, train_t=GATE_TRAIN_T)
    device = init_arena_runtime(config)
    dense_ckpt = _dense_ckpt_path(dry_run=dry_run)
    if not dense_ckpt.exists():
        dense_ckpt = Path(cell.get("phase_a", {}).get("saved_dense_checkpoint", dense_ckpt))
    addr = Path(cell.get("phase_b", {}).get("address_index_checkpoint", ""))
    out_dir = BREAKTHROUGH_OUTPUT / f"gate_systems_{_tag(dry_run)}"
    return run_systems_benchmark(
        config=config,
        train_t=GATE_TRAIN_T,
        dense_ckpt=dense_ckpt,
        address_idx=addr if addr.exists() else None,
        device=device,
        out_dir=out_dir,
        dry_run=dry_run,
    )


def run_gate_aggregate(*, dry_run: bool) -> dict[str, Any]:
    if not AGGREGATE_SCRIPT.exists():
        return {"skipped": True, "reason": "aggregate script missing"}
    out_path = BREAKTHROUGH_OUTPUT / f"results_table_gate_{_tag(dry_run)}.json"
    cmd = [
        sys.executable,
        str(AGGREGATE_SCRIPT),
        "--input",
        str(BREAKTHROUGH_OUTPUT.parent),
        "--output",
        str(out_path),
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
    table = json.loads(out_path.read_text(encoding="utf-8"))
    manifest = {
        "claim": BREAKTHROUGH_CLAIM,
        "gate_t8192": True,
        "dry_run": dry_run,
        "timestamp": _now(),
        "results_table": str(out_path),
        "table": table,
    }
    write_json(BREAKTHROUGH_OUTPUT / f"breakthrough_manifest_gate_{_tag(dry_run)}.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Breakthrough gate suite (T=8192 only)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-systems", action="store_true")
    parser.add_argument("--force-phase-b", action="store_true", help="Retrain Phase B (keep layer 0 if complete)")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    from routing_attention.utils.cuda import configure_cuda_training

    configure_cuda_training({"training": {"cudnn_deterministic": True, "cudnn_benchmark": False}})
    BREAKTHROUGH_OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = BREAKTHROUGH_OUTPUT / f"gate_{_tag(args.dry_run)}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger().addHandler(fh)

    if not args.skip_verify and VERIFY_SCRIPT.exists():
        run_verify()
    preflight(args.dry_run)

    manifest: dict[str, Any] = {
        "dry_run": args.dry_run,
        "train_t": GATE_TRAIN_T,
        "timestamp": _now(),
        "claim": BREAKTHROUGH_CLAIM,
        "steps": {},
    }
    exit_code = 0

    try:
        manifest["steps"]["gate_cell"] = run_gate_cell(
            dry_run=args.dry_run,
            force_phase_b=args.force_phase_b,
        )
    except Exception:
        manifest["steps"]["gate_cell"] = {"error": traceback.format_exc()}
        exit_code = 1

    if exit_code == 0 and not args.skip_systems:
        try:
            manifest["steps"]["systems"] = run_gate_systems(
                dry_run=args.dry_run,
                cell=manifest["steps"]["gate_cell"],
            )
        except Exception:
            manifest["steps"]["systems"] = {"error": traceback.format_exc()}
            exit_code = 1

    if exit_code == 0 and not args.skip_aggregate:
        try:
            manifest["steps"]["aggregate"] = run_gate_aggregate(dry_run=args.dry_run)
        except Exception:
            manifest["steps"]["aggregate"] = {"error": traceback.format_exc()}
            exit_code = 1

    write_json(BREAKTHROUGH_OUTPUT / f"gate_manifest_{_tag(args.dry_run)}.json", manifest)
    logging.getLogger().removeHandler(fh)
    fh.close()
    print(f"\nGate log: {log_path}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
