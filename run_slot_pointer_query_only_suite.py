#!/usr/bin/env python3
"""
slot_pointer @ T=2048 — random 50 quads, query-only value token (lm_token).

No pointer_mlp / pointer_index head. The model sees the queried address at T-1
and is trained to emit the matching value token id from that position's logits.

Usage:
  python run_slot_pointer_query_only_suite.py --dry-run
  python run_slot_pointer_query_only_suite.py
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.production_backends import (
    assert_production_backends_available,
)
from routing_attention.benchmarks.long_context.routing_arena import (
    build_arena_experiment_config,
    init_arena_runtime,
    load_routing_arena_config,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

DEFAULT_CONFIG = ROOT / "configs" / "routing_slot_pointer_t2048_50q_query_only.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "Experiment_7" / "slot_pointer_query_only_t2048_50q"


def preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== slot_pointer query-only preflight ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — training will be slow on CPU.")
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="slot_pointer random 50q — query-only value token (dense_flash)"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-layers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    arena_cfg = load_routing_arena_config(args.config)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = args.n_layers or int(arena_cfg.get("n_layers", 4))

    preflight_info = preflight(args.dry_run)
    config = build_arena_experiment_config(arena_cfg, dry_run=args.dry_run, n_layers=n_layers)

    model_cfg = config.get("model", {})
    bench = config.get("long_context_benchmark", {})
    transformer_cfg = config.get("transformer", {})
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("dense_pretrain_steps")
        or transformer_cfg.get("max_steps", 0)
    )

    print("=== slot_pointer query-only plan ===")
    print(f"  task=slot_pointer  placement=random  quads={bench.get('num_slot_quads', 50)}")
    print(f"  T={train_t}  n_layers={n_layers}  output_head={model_cfg.get('output_head')}")
    print(f"  train_label_mode={bench.get('train_label_mode')}")
    print(f"  steps={steps}  lr={transformer_cfg.get('lr', 3e-4)}")
    print(f"  validate_every={transformer_cfg.get('validate_every')}")
    print(f"  dry_run={args.dry_run}")
    print()

    log_path = args.output_dir / ("run_dry.log" if args.dry_run else "run.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)
    print(f"  log: {log_path}")

    device = init_arena_runtime(config)
    log = logging.getLogger("slot_pointer_query_only.train")
    reset_peak_vram(device)

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
            f"OK dense_flash: gate={gate * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))})"
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

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if args.dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "slot_pointer_query_only_suite",
        "dry_run": args.dry_run,
        "train_context_length": train_t,
        "n_layers": n_layers,
        "output_head": model_cfg.get("output_head"),
        "train_label_mode": bench.get("train_label_mode"),
        "num_slot_quads": bench.get("num_slot_quads"),
        "slot_quad_placement": bench.get("slot_quad_placement"),
        "preflight": preflight_info,
        "result": payload,
        "status": status,
    }
    summary_path = args.output_dir / f"summary_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (args.output_dir / "latest.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  wrote: {summary_path}")

    if status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
