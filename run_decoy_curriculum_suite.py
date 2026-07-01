#!/usr/bin/env python3
"""
Staged decoy curriculum @ T=2048 — teach easy retrieval, then increase decoys.

Stage 0 (0 decoys): bootstrap checkpoints for all variants.
Stages 1 → 2 → 4: continue each variant's own checkpoint on harder decoy count.

Variants: dense_flash, linear, local_window64, local_window256, key_vector_k128 (routing).

Stop rule (full run): first stage where dense >= 90%, linear <= dense - 10pp,
routing within 5pp of dense.

Canonical routing: K=128, LR=3e-4.

Usage:
  python run_decoy_curriculum_suite.py --dry-run
  python run_decoy_curriculum_suite.py
  python run_decoy_curriculum_suite.py --phase bootstrap
  python run_decoy_curriculum_suite.py --phase curriculum
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
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
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    run_attention_baseline,
    run_dense_flash_eval_from_checkpoint,
    run_dense_flash_finetune,
    run_key_vector_k32,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status
from routing_attention.utils.cuda import configure_cuda_training

CONFIG_PATH = ROOT / "configs" / "decoy_curriculum_4L" / "decoy_curriculum_t2048.yaml"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "decoy_curriculum"
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
TRAIN_T = 2048
DECOY_STAGES = [0, 1, 2, 4]
ROUTING_TOP_K = 128
ROUTING_LR = 3e-4

BOOTSTRAP_VARIANTS = ("dense_flash", "linear", "local_window64", "local_window256", "key_vector_k32")

DENSE_MIN = 0.90
LINEAR_GAP_PP = 10.0
ROUTING_MAX_GAP_PP = 5.0


def _official_accuracy(payload: dict) -> float | None:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    return float(acc) if acc is not None else None


def _bench_patch(*, decoys: int, tag: str) -> dict[str, Any]:
    return {
        "benchmark_family": "synthetic",
        "context_lengths": [TRAIN_T],
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
        "benchmark_variant": tag,
        "training_protocol": "two_stage",
    }


def build_stage_config(*, decoys: int, dry_run: bool, bootstrap: bool) -> dict:
    arena = load_routing_arena_config(CONFIG_PATH)
    arena = copy.deepcopy(arena)
    arena["long_context_benchmark"] = _bench_patch(
        decoys=decoys, tag=f"decoy_curriculum_d{decoys}"
    )
    if bootstrap and decoys == 0:
        steps = 20000 if not dry_run else 80
        arena["transformer"]["dense_pretrain_steps"] = steps
        arena["transformer"]["sparse_finetune_steps"] = steps
        arena["key_vector"]["sparse_finetune_steps"] = steps
    config = build_arena_experiment_config(arena, dry_run=dry_run, n_layers=4)
    config.setdefault("key_vector", {})["top_k"] = ROUTING_TOP_K
    config.setdefault("router", {})["top_k"] = ROUTING_TOP_K
    config.setdefault("key_vector", {})["sparse_finetune_lr"] = ROUTING_LR
    return config


def _ckpt_path(stage_dir: Path, variant: str) -> Path:
    return stage_dir / "checkpoints" / f"{variant}.pt"


def preflight(dry_run: bool) -> None:
    configure_cuda_training(
        {"training": {"cudnn_deterministic": True, "cudnn_benchmark": False}}
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    print("=== decoy_curriculum preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    try:
        assert_production_backends_available(list(BOOTSTRAP_VARIANTS))
    except (RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"  dry_run: {dry_run}")
    print(f"  stages: {DECOY_STAGES}")
    print(f"  routing: K={ROUTING_TOP_K} lr={ROUTING_LR}")
    print()


def _gap_signature(dense: float | None, linear: float | None, routing: float | None) -> bool:
    if dense is None or linear is None or routing is None:
        return False
    return (
        dense >= DENSE_MIN
        and linear <= dense - LINEAR_GAP_PP / 100.0
        and routing >= dense - ROUTING_MAX_GAP_PP / 100.0
    )


def run_variant_at_stage(
    *,
    variant: str,
    decoys: int,
    dry_run: bool,
    bootstrap: bool,
    init_ckpt: Path | None,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = build_stage_config(decoys=decoys, dry_run=dry_run, bootstrap=bootstrap)
    save_path = _ckpt_path(out_dir, variant)
    log = logging.getLogger(f"decoy_curriculum.d{decoys}.{variant}")
    reset_peak_vram(device)

    print(f"\n  --- {variant} @ decoys={decoys} init={init_ckpt} ---")
    try:
        if variant == "dense_flash":
            if bootstrap and decoys == 0 and init_ckpt is not None and init_ckpt.exists():
                payload = run_dense_flash_eval_from_checkpoint(
                    config,
                    train_t=TRAIN_T,
                    dense_ckpt=init_ckpt,
                    device=device,
                    log=log,
                )
                save_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(init_ckpt, save_path)
                payload["saved_dense_checkpoint"] = str(save_path)
            else:
                payload = run_dense_flash_finetune(
                    config,
                    train_t=TRAIN_T,
                    dense_ckpt=init_ckpt,
                    device=device,
                    log=log,
                    save_checkpoint_path=save_path,
                )
        elif variant == "key_vector_k32":
            if init_ckpt is None or not init_ckpt.exists():
                raise FileNotFoundError(f"routing requires dense checkpoint: {init_ckpt}")
            payload = run_key_vector_k32(
                config,
                train_t=TRAIN_T,
                dense_ckpt=init_ckpt,
                device=device,
                log=log,
                top_k=ROUTING_TOP_K,
            )
        else:
            payload = run_attention_baseline(
                config,
                variant,
                train_t=TRAIN_T,
                dense_ckpt=init_ckpt,
                device=device,
                log=log,
                save_checkpoint_path=save_path,
            )
        acc = _official_accuracy(payload)
        status = "ok"
    except Exception:
        print(traceback.format_exc())
        payload = {"status": "error"}
        acc = None
        status = "error"
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "variant": variant,
        "decoys": decoys,
        "bootstrap": bootstrap,
        "init_checkpoint": str(init_ckpt) if init_ckpt else None,
        "saved_checkpoint": str(save_path) if save_path.exists() else None,
        "official_accuracy": acc,
        "status": status,
        "result": payload,
    }
    (out_dir / f"{variant}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
    print(f"    {variant}: {acc_s}")
    return summary


def _stage_summary(*, decoys: int, results: dict[str, dict]) -> dict[str, Any]:
    dense = results.get("dense_flash", {}).get("official_accuracy")
    linear = results.get("linear", {}).get("official_accuracy")
    routing = results.get("key_vector_k32", {}).get("official_accuracy")
    local64 = results.get("local_window64", {}).get("official_accuracy")
    local256 = results.get("local_window256", {}).get("official_accuracy")
    return {
        "decoys": decoys,
        "dense_accuracy": dense,
        "linear_accuracy": linear,
        "routing_accuracy": routing,
        "local_window64_accuracy": local64,
        "local_window256_accuracy": local256,
        "dense_minus_linear_pp": (dense - linear) * 100 if dense is not None and linear is not None else None,
        "routing_minus_dense_pp": (routing - dense) * 100 if routing is not None and dense is not None else None,
        "gap_signature": _gap_signature(dense, linear, routing),
        "variants": results,
    }


def run_bootstrap_stage(
    *,
    dry_run: bool,
    device: torch.device,
    canonical_dense: Path,
) -> dict[str, Any]:
    stage_dir = OUTPUT_ROOT / ("bootstrap_dry" if dry_run else "bootstrap")
    stage_dir.mkdir(parents=True, exist_ok=True)
    print("\n########## bootstrap stage decoys=0 ##########")

    results: dict[str, dict] = {}
    dense_ckpt = canonical_dense if canonical_dense.exists() else None

    results["dense_flash"] = run_variant_at_stage(
        variant="dense_flash",
        decoys=0,
        dry_run=dry_run,
        bootstrap=True,
        init_ckpt=dense_ckpt,
        out_dir=stage_dir,
        device=device,
    )
    dense_out = _ckpt_path(stage_dir, "dense_flash")

    for variant in ("linear", "local_window64", "local_window256"):
        results[variant] = run_variant_at_stage(
            variant=variant,
            decoys=0,
            dry_run=dry_run,
            bootstrap=True,
            init_ckpt=None,
            out_dir=stage_dir,
            device=device,
        )

    results["key_vector_k32"] = run_variant_at_stage(
        variant="key_vector_k32",
        decoys=0,
        dry_run=dry_run,
        bootstrap=True,
        init_ckpt=dense_out if dense_out.exists() else dense_ckpt,
        out_dir=stage_dir,
        device=device,
    )

    stage_summary = _stage_summary(decoys=0, results=results)
    (stage_dir / "stage_summary.json").write_text(
        json.dumps(stage_summary, indent=2, default=str), encoding="utf-8"
    )
    return stage_summary


def run_curriculum_stages(
    *,
    dry_run: bool,
    device: torch.device,
    prev_stage_dir: Path,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    prev_dir = prev_stage_dir

    for decoys in [d for d in DECOY_STAGES if d > 0]:
        stage_dir = OUTPUT_ROOT / (f"decoys_{decoys}_dry" if dry_run else f"decoys_{decoys}")
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n########## curriculum stage decoys={decoys} ##########")

        results: dict[str, dict] = {}

        dense_init = _ckpt_path(prev_dir, "dense_flash")
        results["dense_flash"] = run_variant_at_stage(
            variant="dense_flash",
            decoys=decoys,
            dry_run=dry_run,
            bootstrap=False,
            init_ckpt=dense_init if dense_init.exists() else None,
            out_dir=stage_dir,
            device=device,
        )
        dense_out = _ckpt_path(stage_dir, "dense_flash")

        for variant in ("linear", "local_window64", "local_window256"):
            prev_ckpt = _ckpt_path(prev_dir, variant)
            results[variant] = run_variant_at_stage(
                variant=variant,
                decoys=decoys,
                dry_run=dry_run,
                bootstrap=False,
                init_ckpt=prev_ckpt if prev_ckpt.exists() else None,
                out_dir=stage_dir,
                device=device,
            )

        routing_init = dense_out if dense_out.exists() else dense_init
        results["key_vector_k32"] = run_variant_at_stage(
            variant="key_vector_k32",
            decoys=decoys,
            dry_run=dry_run,
            bootstrap=False,
            init_ckpt=routing_init,
            out_dir=stage_dir,
            device=device,
        )

        stage_summary = _stage_summary(decoys=decoys, results=results)
        (stage_dir / "stage_summary.json").write_text(
            json.dumps(stage_summary, indent=2, default=str), encoding="utf-8"
        )
        summaries.append(stage_summary)

        if stage_summary.get("gap_signature") and not dry_run:
            print(f"\n  GAP SIGNATURE at decoys={decoys} — stopping curriculum")
            break

        prev_dir = stage_dir

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged decoy curriculum suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--phase",
        choices=("all", "bootstrap", "curriculum"),
        default="all",
    )
    parser.add_argument("--dense-checkpoint", type=Path, default=DEFAULT_DENSE_CKPT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight(args.dry_run)

    gate: dict[str, Any] = {
        "canonical_seed": CANONICAL_SEED,
        "dry_run": args.dry_run,
        "routing_top_k": ROUTING_TOP_K,
        "routing_lr": ROUTING_LR,
        "dense_checkpoint": str(args.dense_checkpoint),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    bootstrap_summary: dict[str, Any] | None = None
    curriculum_summaries: list[dict] = []

    config = build_stage_config(decoys=0, dry_run=args.dry_run, bootstrap=True)
    device = init_arena_runtime(config)

    if args.phase in ("all", "bootstrap"):
        bootstrap_summary = run_bootstrap_stage(
            dry_run=args.dry_run,
            device=device,
            canonical_dense=args.dense_checkpoint,
        )
        gate["bootstrap"] = bootstrap_summary

    prev_dir = OUTPUT_ROOT / ("bootstrap_dry" if args.dry_run else "bootstrap")
    if args.phase in ("all", "curriculum"):
        if not prev_dir.exists():
            print(f"ERROR: bootstrap dir missing: {prev_dir}")
            sys.exit(1)
        curriculum_summaries = run_curriculum_stages(
            dry_run=args.dry_run,
            device=device,
            prev_stage_dir=prev_dir,
        )
        gate["curriculum_stages"] = curriculum_summaries

    gate["gap_signature_hit"] = any(s.get("gap_signature") for s in curriculum_summaries)
    gate_path = OUTPUT_ROOT / ("success_gate_dry.json" if args.dry_run else "success_gate.json")
    gate_path.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")

    print("\n=== Decoy curriculum summary ===")
    if bootstrap_summary:
        print(
            f"  bootstrap (d=0): dense={bootstrap_summary.get('dense_accuracy')} "
            f"linear={bootstrap_summary.get('linear_accuracy')} "
            f"routing={bootstrap_summary.get('routing_accuracy')} "
            f"local64={bootstrap_summary.get('local_window64_accuracy')} "
            f"local256={bootstrap_summary.get('local_window256_accuracy')}"
        )
    for s in curriculum_summaries:
        print(
            f"  decoys={s['decoys']}: dense={s.get('dense_accuracy')} "
            f"linear={s.get('linear_accuracy')} routing={s.get('routing_accuracy')} "
            f"gap_sig={s.get('gap_signature')}"
        )
    print(f"  gate: {gate_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
