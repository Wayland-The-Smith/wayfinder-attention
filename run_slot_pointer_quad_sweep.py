#!/usr/bin/env python3
"""
Random slot_pointer quad-count sweep — dense_flash @ T=2048.

Default N ∈ {5, 10}. Same protocol as main slot_pointer benchmark:
  random placement, pointer_mlp / value_slots, 4 layers, 30k steps.

Usage:
  python run_slot_pointer_quad_sweep.py
  python run_slot_pointer_quad_sweep.py --quads 5
  python run_slot_pointer_quad_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
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

DEFAULT_CONFIG = ROOT / "configs" / "routing_slot_pointer_quad_sweep.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "Experiment_7" / "slot_pointer_quad_sweep"
DEFAULT_QUADS = [5, 10]


def load_base_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("routing_arena", raw)


def patch_for_n(arena_cfg: dict, num_quads: int) -> dict:
    cfg = copy.deepcopy(arena_cfg)
    bench = dict(cfg.get("long_context_benchmark", {}))
    model = dict(cfg.get("model", {}))
    bench["num_slot_quads"] = int(num_quads)
    bench["holdout_seed"] = int(DEFAULT_HOLDOUT_SEED) + 200 + int(num_quads)
    model["num_pointer_slots"] = int(num_quads)
    cfg["long_context_benchmark"] = bench
    cfg["model"] = model
    cfg["description"] = (
        f"slot_pointer random {num_quads}q @ T=2048 — pointer_mlp (dense_flash)"
    )
    return cfg


def run_one(
    *,
    arena_cfg: dict,
    num_quads: int,
    dry_run: bool,
    output_root: Path,
    device: torch.device,
    log: logging.Logger,
) -> dict:
    patched = patch_for_n(arena_cfg, num_quads)
    config = build_arena_experiment_config(patched, dry_run=dry_run)
    train_t = int(patched["train_context_length"])

    run_dir = output_root / f"N{num_quads:03d}_random_quads_dense_flash"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / ("run_dry.log" if dry_run else "run.log")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)

    steps = int(
        config.get("transformer", {}).get("sparse_finetune_steps")
        or config.get("transformer", {}).get("dense_pretrain_steps", 0)
    )
    print(f"\n########## N={num_quads} random quads ##########")
    print(f"  placement=random  head=pointer_mlp  layers={config.get('model', {}).get('n_layers', 4)}")
    print(f"  steps={steps}  log={log_path}")

    try:
        reset_peak_vram(device)
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
            f"OK N={num_quads}: gate={gate * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))})"
        )
        payload["num_slot_quads"] = num_quads
        payload["status"] = "ok"
        (run_dir / "result.json").write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return payload
    except Exception:
        err = traceback.format_exc()
        print(err)
        return {
            "num_slot_quads": num_quads,
            "status": "error",
            "traceback": err,
        }
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="slot_pointer random quad sweep (dense)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quads", type=int, nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quad_list = args.quads if args.quads else DEFAULT_QUADS
    arena_cfg = load_base_config(args.config)

    preflight = collect_device_info(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    preflight.update(backend_status())
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("=== slot_pointer quad sweep preflight ===")
    for k, v in preflight.items():
        print(f"  {k}: {v}")
    print(f"  quads={quad_list}")
    print()

    init_cfg = build_arena_experiment_config(
        patch_for_n(arena_cfg, quad_list[0]),
        dry_run=args.dry_run,
    )
    device = init_arena_runtime(init_cfg)
    log = logging.getLogger("slot_pointer_quad_sweep")

    results: dict[str, dict] = {}
    errors: list[str] = []
    for n in quad_list:
        payload = run_one(
            arena_cfg=arena_cfg,
            num_quads=n,
            dry_run=args.dry_run,
            output_root=args.output_dir,
            device=device,
            log=log,
        )
        key = f"N{n:03d}"
        results[key] = payload
        if payload.get("status") == "error":
            errors.append(f"{key}: {payload.get('traceback', 'unknown')}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if args.dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "slot_pointer_quad_sweep",
        "dry_run": args.dry_run,
        "train_context_length": int(arena_cfg["train_context_length"]),
        "n_layers": int(arena_cfg.get("n_layers", 4)),
        "placement": "random",
        "output_head": "pointer_mlp",
        "training_steps": int(
            arena_cfg.get("transformer", {}).get("sparse_finetune_steps", 30000)
        ),
        "quad_counts": quad_list,
        "preflight": preflight,
        "reference_gates": {
            "N010_10k_steps_benchmark_suite": 0.13666666666666666,
            "N050_10k_steps_benchmark_suite": 0.04666666666666667,
            "N050_fixed_grid_10k": 0.9833333333333333,
        },
        "results": {
            k: {
                "num_slot_quads": v.get("num_slot_quads"),
                "status": v.get("status", "ok"),
                "eval_official": v.get("eval_official") or v.get("eval"),
                "train_info": {
                    "trained_steps": (v.get("train_info") or {}).get("trained_steps"),
                    "best_holdout": (v.get("train_info") or {}).get("best_holdout"),
                },
            }
            for k, v in results.items()
        },
        "errors": errors,
    }
    summary_path = args.output_dir / f"quad_sweep_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== slot_pointer quad sweep summary ===")
    print(f"  wrote: {summary_path}")
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
