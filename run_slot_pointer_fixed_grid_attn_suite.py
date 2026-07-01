#!/usr/bin/env python3
"""
Train/eval slot_pointer_fixed_grid with dense_flash or linear attention.

Same protocol as benchmark_variants_suite (T=2048, 10k steps, lr=3e-4, 300 holdout).

Usage:
  python run_slot_pointer_fixed_grid_attn_suite.py --attention linear
  python run_slot_pointer_fixed_grid_attn_suite.py --attention dense_flash --n-layers 2
  python run_slot_pointer_fixed_grid_attn_suite.py --dry-run
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
    build_arena_experiment_config,
    init_arena_runtime,
    run_attention_baseline,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status
from run_benchmark_variants_suite import build_variant_arena_cfg, load_suite_config

DEFAULT_SUITE = ROOT / "configs" / "benchmark_variants_suite.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "Experiment_7" / "slot_pointer_fixed_grid_attn"
VARIANT_NAME = "slot_pointer_fixed_grid"
ATTENTION_CHOICES = ("dense_flash", "linear")


def run_one(
    *,
    suite: dict,
    attention: str,
    n_layers: int,
    dry_run: bool,
    output_dir: Path,
    device: torch.device,
    log: logging.Logger,
) -> dict:
    variant = suite["variants"][VARIANT_NAME]
    arena_cfg = build_variant_arena_cfg(suite, VARIANT_NAME, variant, dry_run=dry_run)
    arena_cfg["n_layers"] = int(n_layers)
    arena_cfg["variants"] = [attention]
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run)
    train_t = int(suite["train_context_length"])

    run_key = f"{attention}_L{n_layers}"
    out_dir = output_dir / run_key
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / ("run_dry.log" if dry_run else "run.log")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    print(f"\n########## {VARIANT_NAME} | {run_key} ##########")
    print(f"  attention: {attention}")
    print(f"  n_layers: {n_layers}")
    print(f"  steps: {config.get('transformer', {}).get('sparse_finetune_steps')}")
    print(f"  log: {log_path}")

    try:
        if attention == "dense_flash":
            payload = run_dense_flash_finetune(
                config,
                train_t=train_t,
                dense_ckpt=None,
                device=device,
                log=log,
            )
        elif attention in BASELINE_ATTENTION_VARIANTS:
            payload = run_attention_baseline(
                config,
                attention,
                train_t=train_t,
                dense_ckpt=None,
                device=device,
                log=log,
            )
        else:
            raise ValueError(f"unsupported attention {attention!r}")

        ev = payload.get("eval_official") or payload.get("eval", {})
        gate = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
        print(
            f"OK {run_key}: gate={gate * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))})"
        )
        payload["run_key"] = run_key
        payload["attention"] = attention
        payload["n_layers"] = n_layers
        payload["status"] = "ok"
        summary_path = out_dir / "result.json"
        summary_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {
            "run_key": run_key,
            "attention": attention,
            "n_layers": n_layers,
            "status": "error",
            "traceback": err,
        }
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="slot_pointer_fixed_grid dense vs linear")
    parser.add_argument("--config", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attention", choices=ATTENTION_CHOICES, default="linear")
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument(
        "--sweep-small",
        action="store_true",
        help="Run dense_flash + linear at n_layers 2 and 1 (same protocol)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite_config(args.config)
    if VARIANT_NAME not in suite.get("variants", {}):
        raise SystemExit(f"Missing variant {VARIANT_NAME} in {args.config}")

    preflight = collect_device_info(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    preflight.update(backend_status())
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    log = logging.getLogger("slot_pointer_fixed_grid_attn")
    device = init_arena_runtime(
        build_arena_experiment_config(
            build_variant_arena_cfg(
                suite,
                VARIANT_NAME,
                suite["variants"][VARIANT_NAME],
                dry_run=args.dry_run,
            ),
            dry_run=args.dry_run,
        )
    )

    if args.sweep_small:
        runs = [
            ("dense_flash", 2),
            ("linear", 2),
            ("dense_flash", 1),
            ("linear", 1),
        ]
    else:
        n_layers = int(args.n_layers or suite.get("n_layers", 4))
        runs = [(args.attention, n_layers)]

    results: dict[str, dict] = {}
    errors: list[str] = []
    for attention, n_layers in runs:
        reset_peak_vram(device)
        payload = run_one(
            suite=suite,
            attention=attention,
            n_layers=n_layers,
            dry_run=args.dry_run,
            output_dir=args.output_dir,
            device=device,
            log=log,
        )
        key = payload.get("run_key", f"{attention}_L{n_layers}")
        results[key] = payload
        if payload.get("status") == "error":
            errors.append(f"{key}: {payload.get('traceback', 'unknown')}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if args.dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "slot_pointer_fixed_grid_attn",
        "dry_run": args.dry_run,
        "variant": VARIANT_NAME,
        "preflight": preflight,
        "dense_flash_4L_baseline_gate": 0.9833333333333333,
        "dense_flash_4L_baseline_note": "from benchmark_variants_suite latest.json",
        "results": {
            k: {
                "attention": v.get("attention"),
                "n_layers": v.get("n_layers"),
                "status": v.get("status", "ok"),
                "eval_official": v.get("eval_official") or v.get("eval"),
                "train_info": v.get("train_info"),
            }
            for k, v in results.items()
        },
        "errors": errors,
    }
    summary_path = args.output_dir / f"summary_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== slot_pointer_fixed_grid attention summary ===")
    print(f"  wrote: {summary_path}")
    print(f"  dense 4L reference: 98.33% (295/300)")
    for key, payload in results.items():
        if payload.get("status") == "error":
            print(f"  {key}: ERROR")
            continue
        ev = payload.get("eval_official") or payload.get("eval", {})
        acc = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
        print(
            f"  {key}: gate={acc * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))})"
        )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
