#!/usr/bin/env python3
"""
Gap calibration: addr_val + scatter + decoys (T=1024 / T=2048).

Automated ladder:
  1. Dense-only screen (20k steps) — confirm dense >= 50%.
  2. If dense < 50%: retry dense-only with 2 decoys.
  3. If dense >= 50%: head-to-head dense_flash vs linear vs local_window64 (40k steps).
  4. If dense > 90% on 4 decoys: dense-only with 6 decoys, then head-to-head if still >= 50%.

Pass (task locked): dense >= 60%, dense - linear >= 15 pp, local << dense (local <= dense - 15 pp).

Usage:
  python run_gap_decoys_calibration_suite.py --dry-run
  python run_gap_decoys_calibration_suite.py
  python run_gap_decoys_calibration_suite.py --phase dense_screen --decoys 4
  python run_gap_decoys_calibration_suite.py --phase headtohead --decoys 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    BASELINE_ATTENTION_VARIANTS,
    _resolve_synthetic_bench_cfg,
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    run_attention_baseline,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

DEFAULT_CONFIG = ROOT / "configs" / "routing_gap_decoys_scatter_t2048_4decoys_4L_20k.yaml"


def default_output_root(train_t: int) -> Path:
    return ROOT / "experiments" / "Experiment_7" / f"gap_decoys_scatter_t{train_t}_calibration"

DENSE_SCREEN_MIN = 0.50
DENSE_LOCK_MIN = 0.60
GAP_LOCK_MIN = 0.15
LOCAL_GAP_MIN = 0.15

HEADTOHEAD_VARIANTS = ("dense_flash", "linear", "local_window64")

# Rough train throughput @ T=1024, batch=4, 4L (steps/sec); used for ETA only.
_STEPS_PER_SEC_ESTIMATE = {512: 55.0, 1024: 50.0, 2048: 48.0}


def estimate_variant_minutes(train_t: int, steps: int, n_variants: int) -> float:
    rate = _STEPS_PER_SEC_ESTIMATE.get(int(train_t), 45.0)
    # +~15% for mid-holdout eval @ validate_every = steps/10
    train_sec = (steps / rate) * 1.15
    return (train_sec * n_variants) / 60.0


def print_run_plan(
    *,
    train_t: int,
    phase: str,
    steps: int,
    decoys: int,
    variants: list[str],
    dry_run: bool,
) -> None:
    if dry_run:
        print("=== Run plan (DRY-RUN: 80 steps/variant, full official eval) ===")
    else:
        print("=== Run plan ===")
    print(f"  T={train_t}  decoys={decoys}  phase={phase}  steps={steps}")
    print(f"  variants ({len(variants)}): {', '.join(variants)}")
    if not dry_run:
        mins = estimate_variant_minutes(train_t, steps, len(variants))
        print(f"  estimated wall time: ~{mins:.0f} min ({mins / 60:.1f} h) for training+eval")
        if phase == "full":
            screen = estimate_variant_minutes(train_t, 20000, 1)
            print(f"  (+ dense screen phases: ~{screen:.0f}–{screen * 2:.0f} min before head-to-head)")
    print()


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== gap decoys calibration preflight ===")
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


def build_arena_cfg(
    *,
    train_t: int,
    num_decoys: int,
    steps: int,
    variants: list[str],
    benchmark_variant: str,
    restore_best: bool = True,
) -> dict:
    cfg = {
        "description": (
            f"addr_val {num_decoys} decoys scatter @ T={train_t} — "
            f"{len(variants)} variant(s), {steps} steps"
        ),
        "train_context_length": int(train_t),
        "suite_profile": "full",
        "n_layers": 4,
        "variants": list(variants),
        "dense_finetune_on_task": True,
        "dense_train_from_scratch": True,
        "dense_gate_min": 0,
        "holdout": {"total_samples": 300, "mid_train_samples_per_cell": 10},
        "dense_checkpoint": None,
        "model": {"output_head": "lm_token"},
        "long_context_benchmark": {
            "benchmark_family": "synthetic",
            "context_lengths": [int(train_t)],
            "task_types": ["addr_val"],
            "needle_depths": [0.10, 0.25, 0.50, 0.75, 0.90],
            "suffix_placement": "at_end",
            "scatter_multi_needles": True,
            "num_distractors": int(num_decoys),
            "synthetic_decoy_addrs": int(num_decoys),
            "answer_digit_width": 2,
            "synthetic_conflict_rows": 3,
            "synthetic_hop_count": 1,
            "synthetic_hop_count_min": 1,
            "synthetic_hop_count_max": 1,
            "include_answer_in_suffix": True,
            "train_label_mode": "answer_only",
            "answer_loss_weight": 8.0,
            "benchmark_variant": benchmark_variant,
            "training_protocol": "two_stage",
        },
        "transformer": {
            "sparse_finetune_steps": int(steps),
            "validate_every": max(500, int(steps) // 10),
            "validate_every_min": max(500, int(steps) // 10),
            "log_every": 500,
            "lr": 3.0e-4,
        },
        "routing_attention": {"fair_finetune": True},
        "dense_calibration": {
            "live_metrics": True,
            "early_stop": False,
            "restore_best_checkpoint": restore_best,
            "eval_use_full_holdout": False,
        },
        "dry_run": {
            "sparse_finetune_steps": 80,
            "validate_every": 20,
            "log_every": 10,
            "mid_train_samples_per_cell": 4,
        },
    }
    return cfg


def official_accuracy(payload: dict) -> float | None:
    if not payload or payload.get("status") == "error":
        return None
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy")
    if acc is None:
        acc = ev.get("overall_accuracy")
    return float(acc) if acc is not None else None


def gap_met(dense: float | None, linear: float | None, local: float | None) -> bool:
    if dense is None or linear is None or local is None:
        return False
    return (
        dense >= DENSE_LOCK_MIN
        and (dense - linear) >= GAP_LOCK_MIN
        and (dense - local) >= LOCAL_GAP_MIN
    )


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
    if variant in BASELINE_ATTENTION_VARIANTS:
        return run_attention_baseline(
            config,
            variant,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
        )
    raise ValueError(f"Unsupported variant {variant!r}")


def run_variant_job(
    variant: str,
    *,
    arena_cfg: dict,
    output_dir: Path,
    dry_run: bool,
    preflight_info: dict,
    n_layers: int,
) -> tuple[str, dict, float | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    train_t = int(arena_cfg["train_context_length"])
    holdout_cfg = arena_cfg.get("holdout", {})
    bench = _resolve_synthetic_bench_cfg(config, train_t)
    transformer_cfg = config.get("transformer", {})
    model_cfg = config.get("model", {})
    steps = int(transformer_cfg.get("sparse_finetune_steps") or 0)
    restore_best = bool(config.get("dense_calibration", {}).get("restore_best_checkpoint", True))

    print("=== gap decoys calibration plan ===")
    print(f"  variant={variant}")
    print(f"  task=addr_val  decoys={bench.num_distractors}  scatter={bench.scatter_multi_needles}")
    print(f"  T={train_t}  n_layers={n_layers}  output_head={model_cfg.get('output_head')}")
    print(f"  train_label_mode={bench.train_label_mode}  answer_digits={bench.answer_digit_width}")
    print(f"  steps={steps}  restore_best_checkpoint={restore_best}")
    print(f"  holdout_official_target={holdout_cfg.get('total_samples', 300)}")
    print(f"  dry_run={dry_run}")
    print()

    cfg_path = output_dir / "config_used.yaml"
    cfg_path.write_text(yaml.safe_dump({"routing_arena": arena_cfg}, sort_keys=False), encoding="utf-8")

    log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
    log_path = output_dir / log_name
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)
    print(f"  log: {log_path}")

    device = init_arena_runtime(config)
    log = logging.getLogger("gap_decoys_calibration.train")
    reset_peak_vram(device)
    acc: float | None = None

    try:
        payload = _run_variant(variant, config=config, train_t=train_t, device=device, log=log)
        acc = official_accuracy(payload)
        ev = payload.get("eval_official") or payload.get("eval", {})
        restored = (payload.get("train_info") or {}).get("restored_best_checkpoint", False)
        print(
            f"OK {variant}: official_gate={acc * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))}) "
            f"restored_best={restored}"
        )
        status = "ok"
    except Exception:
        err = traceback.format_exc()
        print(err)
        payload = {"status": "error", "traceback": err}
        status = "error"
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "gap_decoys_calibration",
        "variant": variant,
        "dry_run": dry_run,
        "num_decoys": bench.num_distractors,
        "scatter_multi_needles": bench.scatter_multi_needles,
        "training_steps": steps,
        "restore_best_checkpoint": restore_best,
        "preflight": preflight_info,
        "result": payload,
        "official_accuracy": acc,
        "status": status,
    }
    summary_path = output_dir / f"summary_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_dir / "latest.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  wrote: {summary_path}\n")
    return status, summary, acc


def run_dense_screen(
    *,
    train_t: int,
    num_decoys: int,
    steps: int,
    output_root: Path,
    dry_run: bool,
    n_layers: int,
    preflight_info: dict,
) -> tuple[float | None, Path]:
    label = f"decoys_{num_decoys}_screen_{steps // 1000}k"
    run_dir = output_root / label
    arena_cfg = build_arena_cfg(
        train_t=train_t,
        num_decoys=num_decoys,
        steps=steps,
        variants=["dense_flash"],
        benchmark_variant=f"gap_decoys_scatter_t{train_t}_{label}",
        restore_best=True,
    )
    status, summary, acc = run_variant_job(
        "dense_flash",
        arena_cfg=arena_cfg,
        output_dir=run_dir / "dense_flash",
        dry_run=dry_run,
        preflight_info=preflight_info,
        n_layers=n_layers,
    )
    if status != "ok":
        return None, run_dir
    return acc, run_dir


def run_headtohead(
    *,
    train_t: int,
    num_decoys: int,
    steps: int,
    output_root: Path,
    dry_run: bool,
    n_layers: int,
    preflight_info: dict,
    variants: list[str] | None = None,
) -> dict[str, float | None]:
    label = f"decoys_{num_decoys}_headtohead_{steps // 1000}k"
    run_dir = output_root / label
    active_variants = list(variants or HEADTOHEAD_VARIANTS)
    arena_cfg = build_arena_cfg(
        train_t=train_t,
        num_decoys=num_decoys,
        steps=steps,
        variants=active_variants,
        benchmark_variant=f"gap_decoys_scatter_t{train_t}_{label}",
        restore_best=True,
    )
    accs: dict[str, float | None] = {}
    errors = 0
    summaries: list[dict] = []

    for variant in active_variants:
        print(f"\n########## Head-to-head: {variant} ##########")
        status, summary, acc = run_variant_job(
            variant,
            arena_cfg=arena_cfg,
            output_dir=run_dir / variant,
            dry_run=dry_run,
            preflight_info=preflight_info,
            n_layers=n_layers,
        )
        accs[variant] = acc
        summaries.append(summary)
        if status == "error":
            errors += 1
            print(f"ERROR in {variant}; stopping remaining variants.")
            break

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    dense = accs.get("dense_flash")
    linear = accs.get("linear")
    local = accs.get("local_window64")
    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "gap_decoys_calibration_headtohead",
        "dry_run": dry_run,
        "num_decoys": num_decoys,
        "training_steps": steps,
        "official_accuracy": accs,
        "dense_minus_linear_pp": (dense - linear) if dense is not None and linear is not None else None,
        "dense_minus_local_pp": (dense - local) if dense is not None and local is not None else None,
        "gap_met": gap_met(dense, linear, local),
        "errors": errors,
        "results": summaries,
    }
    combined_path = run_dir / f"combined_{tag}_{stamp}.json"
    combined_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    (run_dir / "combined_latest.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )
    (output_root / "combined_latest.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== Head-to-head summary ===")
    for v in active_variants:
        a = accs.get(v)
        print(f"  {v}: official={a * 100:.2f}%" if a is not None else f"  {v}: (missing)")
    if dense is not None and linear is not None:
        print(f"  dense - linear: {(dense - linear) * 100:.2f} pp")
    if dense is not None and local is not None:
        print(f"  dense - local:  {(dense - local) * 100:.2f} pp")
    print(f"  gap_met (lock task): {combined['gap_met']}")
    print(f"  wrote: {combined_path}\n")
    return accs


def run_full_ladder(
    *,
    train_t: int,
    output_root: Path,
    dry_run: bool,
    n_layers: int,
    always_headtohead: bool = False,
    headtohead_variants: list[str] | None = None,
) -> int:
    """Run automated dense screen → branch → head-to-head ladder."""
    screen_steps = 80 if dry_run else 20000
    headtohead_steps = 80 if dry_run else 40000
    output_root.mkdir(parents=True, exist_ok=True)

    ladder_log: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_context_length": train_t,
        "dry_run": dry_run,
        "phases": [],
        "locked": False,
        "locked_num_decoys": None,
    }

    preflight_info = preflight(["dense_flash"], dry_run)

    # Phase 1: dense screen @ 4 decoys
    print(f"\n########## Phase 1: dense screen (4 decoys, 20k) @ T={train_t} ##########")
    acc4, dir4 = run_dense_screen(
        train_t=train_t,
        num_decoys=4,
        steps=screen_steps,
        output_root=output_root,
        dry_run=dry_run,
        n_layers=n_layers,
        preflight_info=preflight_info,
    )
    ladder_log["phases"].append(
        {"phase": "dense_screen", "num_decoys": 4, "steps": screen_steps, "dense_acc": acc4}
    )

    chosen_decoys = 4
    dense_screen_acc = acc4

    if acc4 is not None and acc4 > 0.90:
        print(f"\nDense {acc4 * 100:.1f}% > 90% — trying harder setting (6 decoys).")
        preflight_info = preflight(["dense_flash"], dry_run)
        acc6, _ = run_dense_screen(
            train_t=train_t,
            num_decoys=6,
            steps=screen_steps,
            output_root=output_root,
            dry_run=dry_run,
            n_layers=n_layers,
            preflight_info=preflight_info,
        )
        ladder_log["phases"].append(
            {"phase": "dense_screen_harder", "num_decoys": 6, "steps": screen_steps, "dense_acc": acc6}
        )
        if acc6 is not None and acc6 >= DENSE_SCREEN_MIN:
            chosen_decoys = 6
            dense_screen_acc = acc6
        elif acc6 is not None and acc6 < DENSE_SCREEN_MIN:
            chosen_decoys = 6
            dense_screen_acc = acc6

    elif acc4 is not None and acc4 < DENSE_SCREEN_MIN:
        print(f"\nDense {acc4 * 100:.1f}% < 50% — retrying with 2 decoys.")
        preflight_info = preflight(["dense_flash"], dry_run)
        acc2, _ = run_dense_screen(
            train_t=train_t,
            num_decoys=2,
            steps=screen_steps,
            output_root=output_root,
            dry_run=dry_run,
            n_layers=n_layers,
            preflight_info=preflight_info,
        )
        ladder_log["phases"].append(
            {"phase": "dense_screen_easier", "num_decoys": 2, "steps": screen_steps, "dense_acc": acc2}
        )
        if acc2 is not None:
            chosen_decoys = 2
            dense_screen_acc = acc2

    if dense_screen_acc is None:
        print("\nSTOP: dense screen failed (no accuracy).")
        ladder_log["stop_reason"] = "dense_screen_failed"
        _write_ladder_log(output_root, ladder_log)
        return 1

    if dense_screen_acc < DENSE_SCREEN_MIN and not always_headtohead:
        print(
            f"\nSTOP: dense screen best={dense_screen_acc} — below {DENSE_SCREEN_MIN * 100:.0f}% threshold."
        )
        ladder_log["stop_reason"] = "dense_below_screen_min"
        _write_ladder_log(output_root, ladder_log)
        return 1

    if dense_screen_acc is not None and dense_screen_acc < DENSE_SCREEN_MIN and always_headtohead:
        print(
            f"\nDense screen {dense_screen_acc * 100:.1f}% < {DENSE_SCREEN_MIN * 100:.0f}% "
            f"— running head-to-head anyway (--always-headtohead)."
        )

    print(
        f"\nDense screen {'passed' if dense_screen_acc >= DENSE_SCREEN_MIN else 'below threshold'} "
        f"({dense_screen_acc * 100:.1f}%) — head-to-head @ {chosen_decoys} decoys, "
        f"{headtohead_steps} steps, T={train_t}."
    )

    h2h_variants = list(headtohead_variants or HEADTOHEAD_VARIANTS)
    preflight_info = preflight(h2h_variants, dry_run)
    accs = run_headtohead(
        train_t=train_t,
        num_decoys=chosen_decoys,
        steps=headtohead_steps,
        output_root=output_root,
        dry_run=dry_run,
        n_layers=n_layers,
        preflight_info=preflight_info,
        variants=h2h_variants,
    )
    ladder_log["phases"].append(
        {
            "phase": "headtohead",
            "num_decoys": chosen_decoys,
            "steps": headtohead_steps,
            "official_accuracy": accs,
        }
    )

    dense = accs.get("dense_flash")
    linear = accs.get("linear")
    local = accs.get("local_window64")
    if gap_met(dense, linear, local):
        ladder_log["locked"] = True
        ladder_log["locked_num_decoys"] = chosen_decoys
        print(f"\nTASK LOCKED: {chosen_decoys} decoys @ T={train_t} scatter addr_val")
    else:
        ladder_log["stop_reason"] = "gap_criteria_not_met"
        print("\nHead-to-head complete but gap lock criteria NOT met:")
        print(f"  need dense >= {DENSE_LOCK_MIN * 100:.0f}%, "
              f"dense-linear >= {GAP_LOCK_MIN * 100:.0f} pp, "
              f"dense-local >= {LOCAL_GAP_MIN * 100:.0f} pp")

    _write_ladder_log(output_root, ladder_log)
    return 0 if ladder_log.get("locked") else 2


def _write_ladder_log(output_root: Path, ladder_log: dict) -> None:
    path = output_root / "ladder_latest.json"
    path.write_text(json.dumps(ladder_log, indent=2, default=str), encoding="utf-8")
    print(f"Ladder log: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gap decoys scatter calibration ladder")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Reference YAML (optional)")
    parser.add_argument("--train-t", type=int, default=2048, help="Train/eval context length")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: experiments/.../gap_decoys_scatter_t{T}_calibration)",
    )
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        choices=list(HEADTOHEAD_VARIANTS),
        help="Head-to-head variants (default: all three). Example: --variants dense_flash linear",
    )
    parser.add_argument(
        "--always-headtohead",
        action="store_true",
        help="Run head-to-head after dense screen even if dense < 50%%",
    )
    parser.add_argument(
        "--phase",
        choices=("full", "dense_screen", "headtohead"),
        default="full",
        help="full=automated ladder; dense_screen or headtohead only",
    )
    parser.add_argument("--decoys", type=int, default=4, help="num_distractors for single-phase runs")
    parser.add_argument("--steps", type=int, default=None, help="Override step count for single-phase runs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = args.output_dir or default_output_root(args.train_t)
    h2h_variants = list(args.variants or HEADTOHEAD_VARIANTS)
    steps_default = 80 if args.dry_run else (20000 if args.phase == "dense_screen" else 40000)
    steps = args.steps or steps_default

    print_run_plan(
        train_t=args.train_t,
        phase=args.phase,
        steps=steps,
        decoys=args.decoys,
        variants=h2h_variants if args.phase == "headtohead" else ["dense_flash"],
        dry_run=args.dry_run,
    )

    if args.phase == "full":
        code = run_full_ladder(
            train_t=args.train_t,
            output_root=output_dir,
            dry_run=args.dry_run,
            n_layers=args.n_layers,
            always_headtohead=args.always_headtohead,
            headtohead_variants=h2h_variants,
        )
        sys.exit(code)

    variants = ["dense_flash"] if args.phase == "dense_screen" else h2h_variants
    preflight_info = preflight(variants, args.dry_run)

    if args.phase == "dense_screen":
        acc, _ = run_dense_screen(
            train_t=args.train_t,
            num_decoys=args.decoys,
            steps=steps,
            output_root=output_dir,
            dry_run=args.dry_run,
            n_layers=args.n_layers,
            preflight_info=preflight_info,
        )
        print(f"Dense official accuracy: {acc * 100:.2f}%" if acc is not None else "Dense run failed.")
        sys.exit(0 if acc is not None and acc >= DENSE_SCREEN_MIN else 1)

    accs = run_headtohead(
        train_t=args.train_t,
        num_decoys=args.decoys,
        steps=steps,
        output_root=output_dir,
        dry_run=args.dry_run,
        n_layers=args.n_layers,
        preflight_info=preflight_info,
        variants=h2h_variants,
    )
    dense = accs.get("dense_flash")
    linear = accs.get("linear")
    local = accs.get("local_window64")
    sys.exit(0 if gap_met(dense, linear, local) else 2)


if __name__ == "__main__":
    main()
