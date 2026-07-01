#!/usr/bin/env python3
"""
Canonical scaling suite — stable recipe + routing parity + decoy/length ladders.

Phase 1: Routing parity sweep on easy pointer_unique @ T=2048 (fixed dense ckpt)
Phase 2: Decoy ladder [1, 2, 4] @ T=2048 — dense / linear / routing three-way
Phase 3: Length ladder [4096, 8192] @ 0 decoys — dense / linear / routing three-way

Stable recipe (locked):
  seed=45, cudnn_deterministic=true, lr_warmup_steps=500, restore_best, 4L, parity harness

Usage:
  python run_canonical_scaling_suite.py --dry-run
  python run_canonical_scaling_suite.py --phase routing
  python run_canonical_scaling_suite.py --phase all
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    run_attention_baseline,
    run_dense_flash_finetune,
    run_key_vector_k32,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.cuda import configure_cuda_training

OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "canonical_scaling"
DEFAULT_DENSE_CKPT = (
    ROOT
    / "experiments"
    / "Experiment_7"
    / "dense_stability_sweep"
    / "replicates"
    / "seed_45"
    / "rep_1"
    / "dense_flash"
    / "dense_flash.pt"
)

CANONICAL_SEED = 45
DENSE_MIN = 0.90
ROUTING_PARITY_MIN = 0.97
ROUTING_MAX_GAP_PP = 5.0
LINEAR_GAP_PP = 10.0

ROUTING_TOP_K = [32, 64, 128]
ROUTING_STEPS = [20000, 40000]
ROUTING_LRS = [1e-4, 3e-4]

DECOY_LEVELS = [1, 2, 4]
LENGTH_LEVELS = [4096, 8192]


def _canonical_arena_base() -> dict[str, Any]:
    return {
        "description": "canonical stable 4L pointer_unique",
        "feasibility_parity": True,
        "holdout_mid_seed_offset": 2,
        "seed": CANONICAL_SEED,
        "suite_profile": "fast",
        "n_layers": 4,
        "dense_finetune_on_task": True,
        "dense_train_from_scratch": True,
        "dense_gate_min": 0,
        "holdout": {"total_samples": 300, "mid_train_samples_per_cell": 10},
        "dense_checkpoint": None,
        "training": {"cudnn_deterministic": True, "cudnn_benchmark": False},
        "transformer": {
            "max_steps": 20000,
            "dense_pretrain_steps": 20000,
            "sparse_finetune_steps": 20000,
            "validate_every": 500,
            "validate_every_min": 500,
            "log_every": 200,
            "lr": 3.0e-4,
            "lr_warmup_steps": 500,
        },
        "dense_calibration": {
            "live_metrics": True,
            "early_stop": False,
            "restore_best_checkpoint": True,
            "eval_use_full_holdout": True,
            "mid_train_samples_per_cell": 4,
            "target_accuracy": 0.90,
        },
        "routing_attention": {"fair_finetune": True},
        "key_vector": {
            "top_k": 128,
            "sparse_finetune_steps": 20000,
            "sparse_finetune_lr": 3.0e-4,
        },
        "router": {"top_k": 128},
        "dry_run": {
            "max_steps": 80,
            "sparse_finetune_steps": 80,
            "validate_every": 20,
            "log_every": 10,
            "mid_train_samples_per_cell": 4,
        },
    }


def _bench_patch(
    *,
    train_t: int,
    decoys: int,
    variant_tag: str,
) -> dict[str, Any]:
    return {
        "benchmark_family": "synthetic",
        "context_lengths": [train_t],
        "task_types": ["pointer_unique"],
        "needle_depths": [0.10, 0.25, 0.50, 0.75, 0.90],
        "suffix_placement": "at_end",
        "scatter_multi_needles": False,
        "synthetic_decoy_keys": decoys,
        "num_distractors": decoys,
        "eval_samples_per_cell": 4,
        "generation_max_attempts": 16,
        "answer_digit_width": 1,
        "answer_loss_weight": 8.0,
        "train_label_mode": "answer_only",
        "include_answer_in_suffix": True,
        "synthetic_hop_count": 1,
        "synthetic_hop_count_min": 1,
        "synthetic_hop_count_max": 1,
        "benchmark_variant": variant_tag,
        "training_protocol": "two_stage",
    }


def build_cell_config(
    *,
    train_t: int,
    decoys: int,
    dry_run: bool,
    routing_steps: int | None = None,
    routing_lr: float | None = None,
    routing_top_k: int | None = None,
    dense_steps: int | None = None,
) -> dict:
    tag = f"pointer_T{train_t}_d{decoys}"
    arena = copy.deepcopy(_canonical_arena_base())
    arena["train_context_length"] = train_t
    arena["description"] = f"canonical {tag}"
    arena["long_context_benchmark"] = _bench_patch(
        train_t=train_t, decoys=decoys, variant_tag=tag
    )
    if dense_steps is not None:
        arena["transformer"]["max_steps"] = dense_steps
        arena["transformer"]["dense_pretrain_steps"] = dense_steps
        arena["transformer"]["sparse_finetune_steps"] = dense_steps
    if routing_steps is not None:
        arena["key_vector"]["sparse_finetune_steps"] = routing_steps
    if routing_lr is not None:
        arena["key_vector"]["sparse_finetune_lr"] = routing_lr
    if routing_top_k is not None:
        arena["key_vector"]["top_k"] = routing_top_k
        arena.setdefault("router", {})["top_k"] = routing_top_k
    return build_arena_experiment_config(arena, dry_run=dry_run, n_layers=4)


def _official_accuracy(payload: dict) -> float | None:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    return float(acc) if acc is not None else None


def preflight(dry_run: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    print("=== canonical_scaling preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    try:
        assert_production_backends_available(["dense_flash", "linear", "key_vector_k32"])
    except (RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"  dry_run: {dry_run}")
    print()


def run_routing_trial(
    *,
    top_k: int,
    steps: int,
    lr: float,
    dense_ckpt: Path,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_t = 2048
    decoys = 0
    config = build_cell_config(
        train_t=train_t,
        decoys=decoys,
        dry_run=dry_run,
        routing_steps=steps,
        routing_lr=lr,
        routing_top_k=top_k,
    )
    label = f"k{top_k}_s{steps}_lr{lr:g}"
    print(f"\n########## routing parity: {label} ##########")

    log_path = out_dir / f"run_{label}.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)
    device = init_arena_runtime(config)
    log = logging.getLogger(f"routing_parity.{label}")
    reset_peak_vram(device)
    try:
        if not dense_ckpt.exists():
            raise FileNotFoundError(f"Dense checkpoint missing: {dense_ckpt}")
        payload = run_key_vector_k32(
            config,
            train_t=train_t,
            dense_ckpt=dense_ckpt,
            device=device,
            log=log,
            top_k=top_k,
        )
        acc = _official_accuracy(payload)
        status = "ok"
    except Exception:
        print(traceback.format_exc())
        payload = {"status": "error"}
        acc = None
        status = "error"
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "label": label,
        "top_k": top_k,
        "steps": steps,
        "lr": lr,
        "official_accuracy": acc,
        "status": status,
        "dense_checkpoint": str(dense_ckpt),
        "result": payload,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
    print(f"  routing acc: {acc_s}")
    return summary


def run_three_way_cell(
    *,
    train_t: int,
    decoys: int,
    out_dir: Path,
    dry_run: bool,
    routing_top_k: int,
    routing_steps: int,
    routing_lr: float,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = build_cell_config(
        train_t=train_t,
        decoys=decoys,
        dry_run=dry_run,
        routing_steps=routing_steps,
        routing_lr=routing_lr,
        routing_top_k=routing_top_k,
    )
    train_t_cfg = int(config.get("data", {}).get("train_context_length", train_t))
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dense_ckpt_path = ckpt_dir / "dense_flash.pt"

    print(f"\n########## three-way T={train_t} decoys={decoys} ##########")
    device = init_arena_runtime(config)
    results: dict[str, Any] = {}
    dense_acc: float | None = None
    dense_ckpt: Path | None = None
    errors = 0

    for variant in ("dense_flash", "linear", "key_vector_k32"):
        print(f"  --- {variant} ---")
        log = logging.getLogger(f"canonical.{variant}")
        reset_peak_vram(device)
        try:
            if variant == "dense_flash":
                payload = run_dense_flash_finetune(
                    config,
                    train_t=train_t_cfg,
                    dense_ckpt=None,
                    device=device,
                    log=log,
                    save_checkpoint_path=dense_ckpt_path,
                )
                dense_acc = _official_accuracy(payload)
                saved = payload.get("saved_dense_checkpoint")
                dense_ckpt = Path(saved) if saved else dense_ckpt_path
            elif variant == "linear":
                payload = run_attention_baseline(
                    config,
                    variant,
                    train_t=train_t_cfg,
                    dense_ckpt=None,
                    device=device,
                    log=log,
                )
            else:
                if dense_ckpt is None or not dense_ckpt.exists():
                    raise FileNotFoundError("routing requires dense checkpoint")
                payload = run_key_vector_k32(
                    config,
                    train_t=train_t_cfg,
                    dense_ckpt=dense_ckpt,
                    device=device,
                    log=log,
                    top_k=routing_top_k,
                )
            acc = _official_accuracy(payload)
            results[variant] = {"accuracy": acc, "status": "ok", "result": payload}
            print(f"    {variant}: {acc * 100:.2f}%" if acc is not None else f"    {variant}: n/a")
        except Exception:
            print(traceback.format_exc())
            results[variant] = {"status": "error"}
            errors += 1
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    linear_acc = (results.get("linear") or {}).get("accuracy")
    routing_acc = (results.get("key_vector_k32") or {}).get("accuracy")
    gap_sig = False
    if dense_acc is not None and linear_acc is not None and routing_acc is not None:
        gap_sig = (
            dense_acc >= DENSE_MIN
            and linear_acc <= dense_acc - LINEAR_GAP_PP / 100.0
            and routing_acc >= dense_acc - ROUTING_MAX_GAP_PP / 100.0
        )

    summary = {
        "train_context_length": train_t,
        "decoys": decoys,
        "dense_accuracy": dense_acc,
        "linear_accuracy": linear_acc,
        "routing_accuracy": routing_acc,
        "dense_minus_linear_pp": (dense_acc - linear_acc) * 100 if dense_acc and linear_acc else None,
        "routing_minus_dense_pp": (routing_acc - dense_acc) * 100 if routing_acc and dense_acc else None,
        "gap_signature": gap_sig,
        "routing_hparams": {"top_k": routing_top_k, "steps": routing_steps, "lr": routing_lr},
        "variants": results,
        "errors": errors,
    }
    (out_dir / "cell_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def _pick_best_routing(trials: list[dict]) -> dict[str, Any]:
    ok = [t for t in trials if t.get("official_accuracy") is not None and t.get("status") == "ok"]
    if not ok:
        return {"top_k": 32, "steps": 20000, "lr": 3e-4, "official_accuracy": None}
    best = max(ok, key=lambda t: float(t["official_accuracy"]))
    return {
        "top_k": best["top_k"],
        "steps": best["steps"],
        "lr": best["lr"],
        "official_accuracy": best["official_accuracy"],
        "label": best["label"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical scaling suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "routing", "decoy", "length"),
        default="all",
    )
    parser.add_argument("--dense-checkpoint", type=Path, default=DEFAULT_DENSE_CKPT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    configure_cuda_training(_canonical_arena_base())
    preflight(args.dry_run)

    gate: dict[str, Any] = {
        "canonical_seed": CANONICAL_SEED,
        "dry_run": args.dry_run,
        "dense_checkpoint": str(args.dense_checkpoint),
    }

    routing_trials: list[dict] = []
    best_routing = {"top_k": 32, "steps": 20000, "lr": 3e-4}

    if args.phase in ("all", "routing"):
        sweep_dir = OUTPUT_ROOT / ("routing_parity_dry" if args.dry_run else "routing_parity")
        for top_k, steps, lr in itertools.product(ROUTING_TOP_K, ROUTING_STEPS, ROUTING_LRS):
            trial_dir = sweep_dir / f"k{top_k}_s{steps}_lr{lr:g}"
            routing_trials.append(
                run_routing_trial(
                    top_k=top_k,
                    steps=steps,
                    lr=lr,
                    dense_ckpt=args.dense_checkpoint,
                    out_dir=trial_dir,
                    dry_run=args.dry_run,
                )
            )
        best_routing = _pick_best_routing(routing_trials)
        gate["routing_sweep"] = routing_trials
        gate["best_routing"] = best_routing
        acc = best_routing.get("official_accuracy")
        print(
            f"\n  best routing: {best_routing.get('label')} "
            f"acc={acc * 100:.2f}%" if acc is not None else ""
        )

    rk = int(best_routing["top_k"])
    rs = int(best_routing["steps"])
    rl = float(best_routing["lr"])

    decoy_results: list[dict] = []
    if args.phase in ("all", "decoy"):
        decoy_root = OUTPUT_ROOT / ("decoy_ladder_dry" if args.dry_run else "decoy_ladder")
        for d in DECOY_LEVELS:
            cell = run_three_way_cell(
                train_t=2048,
                decoys=d,
                out_dir=decoy_root / f"decoys_{d}",
                dry_run=args.dry_run,
                routing_top_k=rk,
                routing_steps=rs,
                routing_lr=rl,
            )
            decoy_results.append(cell)
            if cell.get("gap_signature") and not args.dry_run:
                print(f"\n  GAP SIGNATURE at decoys={d} — stopping decoy ladder")
                break
        gate["decoy_ladder"] = decoy_results

    length_results: list[dict] = []
    if args.phase in ("all", "length"):
        length_root = OUTPUT_ROOT / ("length_ladder_dry" if args.dry_run else "length_ladder")
        for t in LENGTH_LEVELS:
            cell = run_three_way_cell(
                train_t=t,
                decoys=0,
                out_dir=length_root / f"T{t}",
                dry_run=args.dry_run,
                routing_top_k=rk,
                routing_steps=rs,
                routing_lr=rl,
            )
            length_results.append(cell)
            if cell.get("gap_signature") and not args.dry_run:
                print(f"\n  GAP SIGNATURE at T={t} — noted")
        gate["length_ladder"] = length_results

    gate["routing_parity_achieved"] = (
        best_routing.get("official_accuracy") is not None
        and float(best_routing["official_accuracy"]) >= ROUTING_PARITY_MIN
    )
    (OUTPUT_ROOT / "success_gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

    print("\n=== Canonical scaling summary ===")
    if routing_trials:
        print(f"  routing trials: {len(routing_trials)}")
        print(f"  best routing: {best_routing}")
        print(f"  parity (>={ROUTING_PARITY_MIN:.0%}): {gate['routing_parity_achieved']}")
    for cell in decoy_results:
        print(
            f"  decoys={cell['decoys']}: dense={cell.get('dense_accuracy')} "
            f"linear={cell.get('linear_accuracy')} routing={cell.get('routing_accuracy')} "
            f"gap_sig={cell.get('gap_signature')}"
        )
    for cell in length_results:
        print(
            f"  T={cell['train_context_length']}: dense={cell.get('dense_accuracy')} "
            f"linear={cell.get('linear_accuracy')} routing={cell.get('routing_accuracy')} "
            f"gap_sig={cell.get('gap_signature')}"
        )
    print(f"  gate: {OUTPUT_ROOT / 'success_gate.json'}")
    sys.exit(0)


if __name__ == "__main__":
    main()
