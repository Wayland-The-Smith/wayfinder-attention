#!/usr/bin/env python3
"""
mqar_addr_val head-to-head — 4-layer attention baselines, query-only @ T=512.

Trains from scratch, evaluates on the full 300-sample holdout at the **final**
training step (no best-checkpoint restore).

Usage:
  python run_mqar_addr_val_calibration_suite.py --dry-run --all-variants
  python run_mqar_addr_val_calibration_suite.py --all-variants
  python run_mqar_addr_val_calibration_suite.py --variant linear
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

DEFAULT_CONFIG = ROOT / "configs" / "routing_mqar_addr_val_calibration.yaml"
SUPPORTED_VARIANTS = ("dense_flash", "linear")
ALL_VARIANTS = SUPPORTED_VARIANTS


def _default_output_root(arena_cfg: dict) -> Path:
    bench = arena_cfg.get("long_context_benchmark", {})
    variant_label = bench.get("benchmark_variant") or f"n{bench.get('num_kv_pairs', 2)}"
    return ROOT / "experiments" / "Experiment_7" / f"mqar_addr_val_calibration_{variant_label}"


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== mqar_addr_val calibration preflight ===")
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


def run_single_variant(
    variant: str,
    *,
    arena_cfg: dict,
    config: dict,
    train_t: int,
    n_layers: int,
    holdout_cfg: dict,
    bench,
    model_cfg: dict,
    transformer_cfg: dict,
    steps: int,
    output_dir: Path,
    dry_run: bool,
    preflight_info: dict,
) -> tuple[str, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    restore_best = bool(config.get("dense_calibration", {}).get("restore_best_checkpoint", True))

    print("=== mqar_addr_val calibration plan ===")
    print(f"  variant={variant}")
    print(f"  task=mqar_addr_val  N={bench.num_kv_pairs}  queries={bench.num_queries}")
    print(f"  T={train_t}  n_layers={n_layers}  output_head={model_cfg.get('output_head')}")
    print(f"  scatter_multi_needles={bench.scatter_multi_needles}  answer_digits={bench.answer_digit_width}")
    print(f"  train_label_mode={bench.train_label_mode}  include_answer_in_suffix={bench.include_answer_in_suffix}")
    print(f"  steps={steps}  lr={transformer_cfg.get('lr', 3e-4)}")
    print(f"  validate_every={transformer_cfg.get('validate_every')}")
    print(f"  restore_best_checkpoint={restore_best}")
    print(f"  holdout_official_target={holdout_cfg.get('total_samples', 300)}")
    print(f"  dry_run={dry_run}")
    print()

    log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
    log_path = output_dir / log_name
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(fh)
    print(f"  log: {log_path}")

    device = init_arena_runtime(config)
    log = logging.getLogger("mqar_addr_val_calibration.train")
    reset_peak_vram(device)

    try:
        payload = _run_variant(
            variant,
            config=config,
            train_t=train_t,
            device=device,
            log=log,
        )
        ev = payload.get("eval_official") or payload.get("eval", {})
        gate = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
        holdout_n = ev.get("holdout_samples", payload.get("holdout", {}).get("holdout_full_samples"))
        restored = (payload.get("train_info") or {}).get("restored_best_checkpoint", False)
        print(
            f"OK {variant}: official_gate={gate * 100:.2f}% "
            f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
            f"{ev.get('primary_gate_total', ev.get('total'))}) "
            f"[holdout={holdout_n}] restored_best={restored}"
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
        "kind": "mqar_addr_val_calibration_suite",
        "variant": variant,
        "dry_run": dry_run,
        "train_context_length": train_t,
        "n_layers": n_layers,
        "output_head": model_cfg.get("output_head"),
        "train_label_mode": bench.train_label_mode,
        "num_kv_pairs": bench.num_kv_pairs,
        "num_queries": bench.num_queries,
        "answer_digit_width": bench.answer_digit_width,
        "scatter_multi_needles": bench.scatter_multi_needles,
        "training_steps": steps,
        "restore_best_checkpoint": restore_best,
        "holdout_total_samples": holdout_cfg.get("total_samples", 300),
        "preflight": preflight_info,
        "result": payload,
        "status": status,
    }
    summary_path = output_dir / f"summary_{tag}_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (output_dir / "latest.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  wrote: {summary_path}\n")
    return status, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mqar_addr_val head-to-head — dense vs linear @ T=512"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory (default: experiments/.../mqar_addr_val_calibration_<variant>/)",
    )
    parser.add_argument("--n-layers", type=int, default=None, help="Override trunk depth (default: 4)")
    parser.add_argument(
        "--variant",
        choices=SUPPORTED_VARIANTS,
        default="dense_flash",
        help="Single attention baseline (ignored when --all-variants)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Run dense_flash then linear sequentially (same config)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arena_cfg = load_routing_arena_config(args.config)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = args.n_layers or int(arena_cfg.get("n_layers", 4))
    holdout_cfg = arena_cfg.get("holdout", {})
    output_root = args.output_dir or _default_output_root(arena_cfg)

    variants = list(ALL_VARIANTS) if args.all_variants else [args.variant]
    preflight_info = preflight(variants, args.dry_run)
    config = build_arena_experiment_config(arena_cfg, dry_run=args.dry_run, n_layers=n_layers)

    model_cfg = config.get("model", {})
    bench = _resolve_synthetic_bench_cfg(config, train_t)
    transformer_cfg = config.get("transformer", {})
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("dense_pretrain_steps")
        or transformer_cfg.get("max_steps", 0)
    )

    results: list[dict] = []
    errors = 0
    for variant in variants:
        print(f"\n########## Variant: {variant} ##########")
        status, summary = run_single_variant(
            variant,
            arena_cfg=arena_cfg,
            config=config,
            train_t=train_t,
            n_layers=n_layers,
            holdout_cfg=holdout_cfg,
            bench=bench,
            model_cfg=model_cfg,
            transformer_cfg=transformer_cfg,
            steps=steps,
            output_dir=output_root / variant,
            dry_run=args.dry_run,
            preflight_info=preflight_info,
        )
        results.append(summary)
        if status == "error":
            errors += 1
            if args.all_variants:
                print(f"ERROR in {variant}; stopping remaining variants.")
                break

    if len(variants) > 1:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        tag = "dry_run" if args.dry_run else "full"
        combined = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": "mqar_addr_val_calibration_suite_combined",
            "dry_run": args.dry_run,
            "variants": variants,
            "num_kv_pairs": bench.num_kv_pairs,
            "training_steps": steps,
            "restore_best_checkpoint": bool(
                config.get("dense_calibration", {}).get("restore_best_checkpoint", True)
            ),
            "results": results,
            "errors": errors,
        }
        combined_path = output_root / f"combined_{tag}_{stamp}.json"
        combined_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
        (output_root / "combined_latest.json").write_text(
            json.dumps(combined, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"=== Combined summary ===")
        print(f"  wrote: {combined_path}")
        for item in results:
            v = item["variant"]
            if item.get("status") == "ok":
                ev = (item.get("result") or {}).get("eval_official") or (item.get("result") or {}).get("eval", {})
                acc = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
                print(f"  {v}: official_gate={acc * 100:.2f}%")
            else:
                print(f"  {v}: ERROR")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
