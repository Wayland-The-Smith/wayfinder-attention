#!/usr/bin/env python3
"""
4L three-way NIAH diagnostic — reproduce feasibility dense ceiling, then compare routing.

Order:
  1. dense_flash   — from scratch (feasibility-matched 4L @ T=2048, 20k)
  2. linear        — from scratch (same harness)
  3. key_vector_k32 — routing sparse top-k, finetuned from dense checkpoint

Gates (full run):
  Step 1: dense official acc >= 90%
  Step 2: |routing - dense| <= 5 pp (if dense passed)

Usage:
  python scripts/verify_niah_three_way_4L.py
  python run_niah_three_way_4L_suite.py --dry-run
  python run_niah_three_way_4L_suite.py
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
    run_dense_flash_eval_from_checkpoint,
    run_dense_flash_finetune,
    run_key_vector_k32,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

CONFIG_PATH = ROOT / "configs" / "niah_three_way_4L" / "niah_pointer_unique_t2048.yaml"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "niah_three_way_4L"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_niah_three_way_4L.py"

VARIANT_ORDER = ("dense_flash", "linear", "key_vector_k32")
DENSE_MIN = 0.90
ROUTING_MAX_GAP_PP = 5.0


def preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== niah_three_way_4L preflight ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — training will be slow on CPU.")
    try:
        assert_production_backends_available(list(VARIANT_ORDER))
    except (RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print()
    return info


def run_verify() -> None:
    print("=== Dataset verification ===")
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print()


def _official_accuracy(payload: dict) -> float | None:
    ev = payload.get("eval_official") or payload.get("eval") or {}
    acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
    return float(acc) if acc is not None else None


def _run_variant(
    variant: str,
    *,
    config: dict,
    train_t: int,
    device: torch.device,
    log: logging.Logger,
    dense_ckpt: Path | None,
    save_dense_ckpt: Path | None,
) -> dict:
    if variant == "dense_flash":
        return run_dense_flash_finetune(
            config,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
            save_checkpoint_path=save_dense_ckpt,
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
    if variant == "key_vector_k32":
        if dense_ckpt is None or not dense_ckpt.exists():
            raise FileNotFoundError(
                f"key_vector_k32 requires dense checkpoint; missing {dense_ckpt}"
            )
        return run_key_vector_k32(
            config,
            train_t=train_t,
            dense_ckpt=dense_ckpt,
            device=device,
            log=log,
            top_k=int(config.get("key_vector", {}).get("top_k") or config.get("router", {}).get("top_k", 128)),
        )
    raise ValueError(f"Unknown variant {variant!r}")


def run_suite(
    *,
    dry_run: bool,
    skip_verify: bool,
    dense_checkpoint: Path | None = None,
) -> int:
    if not skip_verify:
        run_verify()

    arena_cfg = load_routing_arena_config(CONFIG_PATH)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = int(arena_cfg.get("n_layers", 4))
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    bench = _resolve_synthetic_bench_cfg(config, train_t)
    steps = int(config.get("transformer", {}).get("sparse_finetune_steps") or 0)
    cal = config.get("dense_calibration", {})

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ckpt_dir = OUTPUT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dense_ckpt_path = ckpt_dir / ("dense_flash_dry.pt" if dry_run else "dense_flash_T2048.pt")

    print("=== Experiment plan ===")
    print(f"  task=pointer_unique  T={train_t}  n_layers={n_layers}  steps={steps}")
    print(f"  scatter={bench.scatter_multi_needles}  decoys={bench.synthetic_decoy_keys}")
    print(f"  restore_best={cal.get('restore_best_checkpoint')}")
    print(f"  variants={VARIANT_ORDER}")
    print(f"  output={OUTPUT_ROOT}")
    if dry_run:
        print("  mode=dry-run (smoke only)")
    print()

    suite_log = OUTPUT_ROOT / ("full_run_suite_dry.log" if dry_run else "full_run_suite.log")
    suite_fh = logging.FileHandler(suite_log, encoding="utf-8")
    suite_fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(suite_fh)

    device = init_arena_runtime(config)
    errors = 0
    results: dict[str, dict] = {}
    dense_acc: float | None = None
    dense_ckpt: Path | None = Path(dense_checkpoint) if dense_checkpoint else None
    dense_eval_only = dense_ckpt is not None and dense_ckpt.exists()

    for variant in VARIANT_ORDER:
        print(f"\n########## {variant} ##########")
        if variant == "dense_flash" and dense_eval_only:
            print(f"  dense eval-only from checkpoint: {dense_ckpt}")
        elif variant == "key_vector_k32" and dense_ckpt is None:
            print(f"SKIP {variant}: no dense checkpoint available")
            results[variant] = {"status": "skipped", "reason": "missing_dense_checkpoint"}
            continue

        variant_dir = OUTPUT_ROOT / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
        log_path = variant_dir / log_name
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
        print(f"  log: {log_path}")

        log = logging.getLogger(f"niah_three_way_4L.{variant}")
        reset_peak_vram(device)
        save_dense = dense_ckpt_path if variant == "dense_flash" else None

        try:
            if variant == "dense_flash" and dense_eval_only:
                payload = run_dense_flash_eval_from_checkpoint(
                    config,
                    train_t=train_t,
                    dense_ckpt=dense_ckpt,
                    device=device,
                    log=log,
                )
            else:
                payload = _run_variant(
                    variant,
                    config=config,
                    train_t=train_t,
                    device=device,
                    log=log,
                    dense_ckpt=dense_ckpt if variant == "key_vector_k32" else None,
                    save_dense_ckpt=save_dense,
                )
            acc = _official_accuracy(payload)
            acc_s = f"{acc * 100:.2f}%" if acc is not None else "n/a"
            print(f"OK {variant}: official_acc={acc_s}")
            if variant == "dense_flash":
                dense_acc = acc
                if not dense_eval_only:
                    saved = payload.get("saved_dense_checkpoint")
                    if saved and Path(saved).exists():
                        dense_ckpt = Path(saved)
                    elif dense_ckpt_path.exists():
                        dense_ckpt = dense_ckpt_path
                elif dense_ckpt is None and dense_checkpoint:
                    dense_ckpt = Path(dense_checkpoint)
            status = "ok"
        except Exception:
            err = traceback.format_exc()
            print(err)
            payload = {"status": "error", "traceback": err}
            status = "error"
            errors += 1
        finally:
            logging.getLogger().removeHandler(fh)
            fh.close()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = "dry_run" if dry_run else "full"
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": "niah_three_way_4L",
            "variant": variant,
            "dry_run": dry_run,
            "train_context_length": train_t,
            "n_layers": n_layers,
            "training_steps": steps,
            "result": payload,
            "status": status,
        }
        if isinstance(payload, dict) and status == "ok":
            summary["official_accuracy"] = _official_accuracy(payload)
        (variant_dir / "latest.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        (variant_dir / f"summary_{tag}_{stamp}.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        results[variant] = summary

    linear_acc = results.get("linear", {}).get("official_accuracy")
    routing_acc = results.get("key_vector_k32", {}).get("official_accuracy")

    gate = {
        "dense_accuracy": dense_acc,
        "linear_accuracy": linear_acc,
        "routing_accuracy": routing_acc,
        "dense_min": DENSE_MIN,
        "step1_dense_passed": dense_acc is not None and dense_acc >= DENSE_MIN,
        "routing_max_gap_pp": ROUTING_MAX_GAP_PP,
    }
    if dense_acc is not None and routing_acc is not None:
        gate["routing_minus_dense_pp"] = (routing_acc - dense_acc) * 100.0
        gate["step2_routing_near_dense"] = abs(routing_acc - dense_acc) * 100.0 <= ROUTING_MAX_GAP_PP
    else:
        gate["routing_minus_dense_pp"] = None
        gate["step2_routing_near_dense"] = None

    if linear_acc is not None and dense_acc is not None:
        gate["dense_minus_linear_pp"] = (dense_acc - linear_acc) * 100.0

    (OUTPUT_ROOT / "success_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    combined = {k: v for k, v in results.items()}
    combined["gate"] = gate
    (OUTPUT_ROOT / "combined_latest.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    logging.getLogger().removeHandler(suite_fh)
    suite_fh.close()

    print(f"\n=== Suite finished ({errors} errors) ===")
    print(f"  dense:   {dense_acc * 100:.2f}%" if dense_acc is not None else "  dense:   n/a")
    print(f"  linear:  {linear_acc * 100:.2f}%" if linear_acc is not None else "  linear:  n/a")
    print(f"  routing: {routing_acc * 100:.2f}%" if routing_acc is not None else "  routing: n/a")
    print(f"  step1 (dense>={DENSE_MIN:.0%}): {gate['step1_dense_passed']}")
    if gate.get("routing_minus_dense_pp") is not None:
        print(
            f"  step2 (routing within {ROUTING_MAX_GAP_PP:.0f}pp of dense): "
            f"{gate['step2_routing_near_dense']} ({gate['routing_minus_dense_pp']:+.1f} pp)"
        )
    print(f"  gate: {OUTPUT_ROOT / 'success_gate.json'}")
    print(f"  log:  {suite_log}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="4L three-way NIAH diagnostic suite")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--dense-checkpoint",
        type=Path,
        default=None,
        help="Use saved dense checkpoint (eval-only for dense; init for routing)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    preflight(args.dry_run)
    sys.exit(
        run_suite(
            dry_run=args.dry_run,
            skip_verify=args.skip_verify,
            dense_checkpoint=args.dense_checkpoint,
        )
    )


if __name__ == "__main__":
    main()
