#!/usr/bin/env python3
"""
Slot-pointer benchmark — dense vs linear (pointer-index head).

Task
----
- Token-native ``slot_pointer`` @ T=2048, 50 contiguous ``[addr][;][value][,]`` quads.
- 129-token vocab (100 semantic + delimiters + hay tokens).
- Question = queried address token at index T−1; **no answer in the input**.
- Supervision: ``model.output_head=pointer_index`` — CE over sequence positions
  0..T−2 from the question hidden state (Hit@1 index accuracy at eval).

Training (this runner)
------------------------
- Default variant: ``dense_flash`` (full dense SDPA / Flash when available).
- Trains from scratch on the task (no external dense checkpoint required).
- Full run: 10k steps @ lr=3e-4 (see config ``transformer.sparse_finetune_steps``).
- Mid-train holdout: stratified subsample; official eval on full 300-sample grid.

Usage
-----
  python run_slot_pointer_suite.py --dry-run
  python run_slot_pointer_suite.py
  python run_slot_pointer_suite.py --variants dense_flash linear
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
    run_attention_baseline,
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

DEFAULT_CONFIG = ROOT / "configs" / "routing_slot_pointer_t2048_50q.yaml"
DEFAULT_OUTPUT = ROOT / "experiments" / "Experiment_7" / "slot_pointer_t2048_50q"
SUPPORTED_VARIANTS = ("dense_flash", "linear")

logger = logging.getLogger("slot_pointer_suite")


def preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== Slot-pointer preflight ===")
    for key, value in info.items():
        print(f"  {key}: {value}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — T=2048 training will be very slow on CPU.")
    if not info.get("fla_linear"):
        print("ERROR: flash-linear-attention (fla) required for linear baseline.")
        sys.exit(1)
    try:
        assert_production_backends_available()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print()
    return info


def _print_plan(arena_cfg: dict, config: dict, *, dry_run: bool, variants: list[str]) -> None:
    train_t = int(arena_cfg["train_context_length"])
    model_cfg = config.get("model", {})
    transformer_cfg = config.get("transformer", {})
    bench = config.get("long_context_benchmark", {})
    holdout = config.get("holdout", arena_cfg.get("holdout", {}))

    print("=== Slot-pointer plan ===")
    print(f"  task=slot_pointer  T={train_t}  quads={bench.get('num_slot_quads', 50)}")
    print(f"  output_head={model_cfg.get('output_head', 'lm_token')}  vocab_size={model_cfg.get('vocab_size')}")
    print(f"  n_layers={model_cfg.get('n_layers')}  variants={variants}")
    print(f"  train_from_scratch={bool(arena_cfg.get('dense_train_from_scratch', True))}")
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("max_steps", 0)
    )
    print(f"  steps={steps}  lr={transformer_cfg.get('lr', 3e-4)}")
    print(f"  validate_every={transformer_cfg.get('validate_every')}  log_every={transformer_cfg.get('log_every')}")
    print(f"  holdout_official={holdout.get('total_samples', 'per-cell default')}")
    print(f"  mid_train_per_cell={holdout.get('mid_train_samples_per_cell', '?')}")
    print(f"  dry_run={dry_run}")
    if dry_run:
        dry = arena_cfg.get("dry_run", {})
        print(f"  dry sparse steps={dry.get('sparse_finetune_steps', 40)}")
        print(f"  dry validate_every={dry.get('validate_every', 10)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Slot-pointer dense/linear benchmark @ T=2048")
    parser.add_argument("--dry-run", action="store_true", help="Short smoke run (~40 train steps)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["dense_flash"],
        choices=list(SUPPORTED_VARIANTS),
        help="Attention variants to run (default: dense_flash only)",
    )
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    arena_cfg = load_routing_arena_config(args.config)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = args.n_layers or int(arena_cfg.get("n_layers", 4))

    preflight_info = preflight(args.dry_run)
    config = build_arena_experiment_config(arena_cfg, dry_run=args.dry_run, n_layers=n_layers)
    _print_plan(arena_cfg, config, dry_run=args.dry_run, variants=list(args.variants))

    out_dir = args.output_dir
    log_path = out_dir / ("run_dry.log" if args.dry_run else "run.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(file_handler)
    print(f"  logging to {log_path}")

    device = init_arena_runtime(config)
    log = logging.getLogger("slot_pointer.train")

    results: dict[str, dict] = {}
    errors: list[str] = []

    for variant in args.variants:
        print(f"\n########## Variant: {variant} ##########")
        reset_peak_vram(device)
        try:
            if variant == "dense_flash":
                payload = run_dense_flash_finetune(
                    config,
                    train_t=train_t,
                    dense_ckpt=None,
                    device=device,
                    log=log,
                )
            elif variant == "linear":
                payload = run_attention_baseline(
                    config,
                    variant,
                    train_t=train_t,
                    dense_ckpt=None,
                    device=device,
                    log=log,
                )
            else:
                raise ValueError(f"Unsupported variant: {variant}")

            ev = payload.get("eval_official") or payload.get("eval", {})
            gate = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
            train_info = payload.get("train_info", {})
            print(
                f"OK {variant}: official_gate={float(gate or 0) * 100:.2f}% "
                f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
                f"{ev.get('primary_gate_total', ev.get('total'))}) "
                f"trained_steps={train_info.get('trained_steps')} "
                f"[holdout={ev.get('holdout_samples', '?')}]"
            )
            results[variant] = payload
        except Exception:
            err = traceback.format_exc()
            errors.append(f"{variant}: {err}")
            results[variant] = {"variant": variant, "status": "error", "traceback": err}
            print(err)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if args.dry_run else "full"
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "slot_pointer_suite",
        "dry_run": args.dry_run,
        "train_context_length": train_t,
        "n_layers": n_layers,
        "output_head": config.get("model", {}).get("output_head"),
        "vocab_size": config.get("model", {}).get("vocab_size"),
        "num_slot_quads": config.get("long_context_benchmark", {}).get("num_slot_quads"),
        "variants": list(args.variants),
        "preflight": preflight_info,
        "results": results,
        "errors": errors,
    }
    summary_path = out_dir / f"slot_pointer_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Slot-pointer summary ===")
    print(f"  wrote: {summary_path}")
    for variant, payload in results.items():
        if payload.get("status") == "error":
            print(f"  {variant}: ERROR")
            continue
        ev = payload.get("eval_official") or payload.get("eval", {})
        acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0))
        print(
            f"  {variant}: official_gate={float(acc) * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))})"
        )

    if errors:
        print(f"  errors={len(errors)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
