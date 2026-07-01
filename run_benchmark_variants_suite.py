#!/usr/bin/env python3
"""
Run the four benchmark variants (slot_pointer easements + pointer_unique @ T=2048).

Variants:
  1. slot_pointer_10q           — 10 quads, pointer_mlp / value_slots
  2. slot_pointer_fixed_grid    — 50 quads, fixed grid placement
  3. slot_pointer_unique_semantics — no stray semantic tokens outside quads
  4. pointer_unique_t2048       — L0 NIAH, 0 decoys, query-only suffix (no answer in input)

Usage:
  python run_benchmark_variants_suite.py
  python run_benchmark_variants_suite.py --variant slot_pointer_10q
  python run_benchmark_variants_suite.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.config import DEFAULT_HOLDOUT_SEED
from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    init_arena_runtime,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

DEFAULT_SUITE = ROOT / "configs" / "benchmark_variants_suite.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "Experiment_7" / "benchmark_variants_suite"


def load_suite_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("benchmark_variants_suite", raw)


def build_variant_arena_cfg(suite: dict, variant_name: str, variant: dict, *, dry_run: bool) -> dict:
    slot_bench = dict(suite.get("slot_pointer_defaults", {}))
    slot_model = dict(suite.get("slot_pointer_model_defaults", {}))
    bench_patch = {**slot_bench, **variant.get("long_context_benchmark", {})}
    model_patch = {**slot_model, **variant.get("model", {})}

    if bench_patch.get("task_types") == ["pointer_unique"]:
        model_patch.setdefault("output_head", "lm_token")
        model_patch.setdefault("vocab_size", 128)
    else:
        model_patch.setdefault("num_pointer_slots", bench_patch.get("num_slot_quads", 50))

    offset = int(variant.get("holdout_seed_offset", 0))
    bench_patch["holdout_seed"] = int(DEFAULT_HOLDOUT_SEED) + offset

    arena = {
        "description": variant.get("description", variant_name),
        "train_context_length": int(suite["train_context_length"]),
        "n_layers": int(suite.get("n_layers", 4)),
        "suite_profile": suite.get("suite_profile", "full"),
        "variants": ["dense_flash"],
        "dense_finetune_on_task": True,
        "dense_train_from_scratch": True,
        "dense_gate_min": 0,
        "model": model_patch,
        "holdout": dict(suite.get("holdout", {})),
        "long_context_benchmark": bench_patch,
        "transformer": dict(suite.get("transformer", {})),
        "dense_calibration": dict(suite.get("dense_calibration", {})),
        "routing_attention": {"fair_finetune": True},
    }
    if dry_run:
        arena["dry_run"] = {
            "sparse_finetune_steps": 40,
            "validate_every": 10,
            "validate_every_min": 10,
            "log_every": 10,
            "mid_train_samples_per_cell": 4,
        }
    return arena


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark variant suite @ T=2048")
    parser.add_argument("--config", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", type=str, default=None, help="Run one variant only")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite_config(args.config)
    variants: dict = suite.get("variants", {})
    if not variants:
        raise SystemExit("No variants defined in config")

    names = [args.variant] if args.variant else list(variants.keys())
    for name in names:
        if name not in variants:
            raise SystemExit(f"Unknown variant: {name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preflight = collect_device_info(device)
    preflight.update(backend_status())
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    log = logging.getLogger("benchmark_variants.train")
    results: dict[str, dict] = {}
    errors: list[str] = []
    device = init_arena_runtime(
        build_arena_experiment_config(
            build_variant_arena_cfg(suite, names[0], variants[names[0]], dry_run=args.dry_run),
            dry_run=args.dry_run,
        )
    )

    for name in names:
        print(f"\n########## Variant: {name} ##########")
        reset_peak_vram(device)
        variant = variants[name]
        arena_cfg = build_variant_arena_cfg(suite, name, variant, dry_run=args.dry_run)
        config = build_arena_experiment_config(arena_cfg, dry_run=args.dry_run)
        train_t = int(suite["train_context_length"])

        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / ("run_dry.log" if args.dry_run else "run.log")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)

        print(f"  desc: {variant.get('description', '')}")
        print(f"  tasks: {config.get('long_context_benchmark', {}).get('task_types')}")
        print(f"  output_head: {config.get('model', {}).get('output_head')}")
        print(f"  steps: {config.get('transformer', {}).get('sparse_finetune_steps')}")
        print(f"  log: {log_path}")

        try:
            payload = run_dense_flash_finetune(
                config,
                train_t=train_t,
                dense_ckpt=None,
                device=device,
                log=log,
            )
            ev = payload.get("eval_official") or payload.get("eval", {})
            gate = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
            print(
                f"OK {name}: gate={gate * 100:.2f}% "
                f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
                f"{ev.get('primary_gate_total', ev.get('total'))})"
            )
            results[name] = payload
        except Exception:
            err = traceback.format_exc()
            errors.append(f"{name}: {err}")
            results[name] = {"status": "error", "traceback": err}
            print(err)
        finally:
            logging.getLogger().removeHandler(fh)
            fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if args.dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "benchmark_variants_suite",
        "dry_run": args.dry_run,
        "variants_run": names,
        "preflight": preflight,
        "results": {
            k: {
                "eval_official": v.get("eval_official") or v.get("eval"),
                "train_info": v.get("train_info"),
                "status": v.get("status", "ok"),
            }
            for k, v in results.items()
        },
        "errors": errors,
    }
    summary_path = args.output_dir / f"variants_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Variants summary ===")
    print(f"  wrote: {summary_path}")
    for name in names:
        payload = results.get(name, {})
        if payload.get("status") == "error":
            print(f"  {name}: ERROR")
            continue
        ev = payload.get("eval_official") or payload.get("eval", {})
        acc = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
        print(
            f"  {name}: gate={acc * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))})"
        )
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
