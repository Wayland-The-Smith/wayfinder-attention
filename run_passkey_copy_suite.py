#!/usr/bin/env python3
"""
passkey_copy NIAH @ T=4096 — 6L dense (or linear) training with held-out official eval.

Training uses procedural seed=42; holdout uses holdout_seed (disjoint — never in training).
Full run evaluates the final checkpoint on the full 300-sample holdout.

Usage:
  python scripts/verify_passkey_copy.py
  python run_passkey_copy_suite.py --dry-run
  python run_passkey_copy_suite.py --dry-run --variants dense_flash
  python run_passkey_copy_suite.py --variants dense_flash,linear
  python run_passkey_copy_suite.py
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
    run_dense_flash_finetune,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info, reset_peak_vram
from routing_attention.models.fast_attention import backend_status

CONFIG_PATH = ROOT / "configs" / "routing_passkey_copy_t4096_6L_40k.yaml"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "passkey_copy_t4096_6L_40k"
VERIFY_SCRIPT = ROOT / "scripts" / "verify_passkey_copy.py"
SUPPORTED_VARIANTS = ("dense_flash", "linear")


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== passkey_copy preflight ===")
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


def run_verify() -> None:
    print(f"=== Dataset verification ({VERIFY_SCRIPT.name}) ===")
    proc = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT)},
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    print()


def _parse_variants(raw: str | None, arena_cfg: dict) -> list[str]:
    if raw:
        variants = [v.strip() for v in raw.split(",") if v.strip()]
    else:
        variants = list(arena_cfg.get("variants") or ["dense_flash"])
    bad = [v for v in variants if v not in SUPPORTED_VARIANTS]
    if bad:
        raise ValueError(f"Unsupported variants {bad}; choose from {SUPPORTED_VARIANTS}")
    return variants


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
    if variant == "linear":
        return run_attention_baseline(
            config,
            variant,
            train_t=train_t,
            dense_ckpt=None,
            device=device,
            log=log,
        )
    raise ValueError(f"Unsupported variant {variant!r}")


def run_suite(
    *,
    dry_run: bool,
    skip_verify: bool,
    variants: list[str],
    config_path: Path,
    output_root: Path,
) -> int:
    if not skip_verify:
        run_verify()

    arena_cfg = load_routing_arena_config(config_path)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = int(arena_cfg.get("n_layers", 6))
    output_root.mkdir(parents=True, exist_ok=True)

    preflight_info = preflight(variants, dry_run)
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)

    cal = config.setdefault("dense_calibration", {})
    if dry_run:
        cal["eval_use_full_holdout"] = False
    else:
        cal.setdefault("eval_use_full_holdout", True)
    cal.setdefault("restore_best_checkpoint", False)

    bench = _resolve_synthetic_bench_cfg(config, train_t)
    transformer_cfg = config.get("transformer", {})
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("dense_pretrain_steps")
        or 0
    )

    print("=== Experiment plan ===")
    print(f"  task=passkey_copy  T={train_t}  scatter={bench.scatter_multi_needles}")
    print(f"  answer_digit_width={bench.answer_digit_width}  layers={n_layers}")
    print(f"  steps={steps}  variants={variants}  dry_run={dry_run}")
    print(f"  train_seed={bench.seed}  holdout_seed={bench.holdout_seed}")
    print(f"  restore_best_checkpoint={cal.get('restore_best_checkpoint')}")
    print(f"  eval_use_full_holdout={cal.get('eval_use_full_holdout')}")
    print(f"  output={output_root}")
    if dry_run:
        print("  dry-run eval: mid holdout subset only (not full 300)")
    else:
        print("  full-run eval: official 300-sample holdout on final weights")
    if not dry_run:
        print("  estimated wall time: ~20–30 min per variant @ T=4096 6L 40k")
    print()

    results: list[dict] = []
    errors = 0
    for variant in variants:
        print(f"\n########## passkey_copy / {variant} ##########")
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
        log_path = variant_dir / log_name
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
        print(f"  log: {log_path}")

        device = init_arena_runtime(config)
        log = logging.getLogger(f"passkey_copy.{variant}")
        reset_peak_vram(device)
        try:
            payload = _run_variant(variant, config=config, train_t=train_t, device=device, log=log)
            ev = payload.get("eval_official") or payload.get("eval", {})
            acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
            acc_f = float(acc) if acc is not None else None
            restored = (payload.get("train_info") or {}).get("restored_best_checkpoint")
            holdout_meta = (payload.get("holdout") or {})
            subset = ev.get("eval_subset", "unknown")
            if acc_f is not None:
                acc_str = f"{acc_f * 100:.2f}%"
            else:
                acc_str = "n/a (official skipped)"
            print(
                f"OK {variant}: eval_gate={acc_str} "
                f"subset={subset} "
                f"restored_best={restored} "
                f"holdout_official={holdout_meta.get('holdout_full_samples')}"
            )
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
            "kind": "passkey_copy",
            "variant": variant,
            "dry_run": dry_run,
            "train_context_length": train_t,
            "n_layers": n_layers,
            "answer_digit_width": bench.answer_digit_width,
            "scatter_multi_needles": bench.scatter_multi_needles,
            "training_steps": steps,
            "train_seed": bench.seed,
            "holdout_seed": bench.holdout_seed,
            "restore_best_checkpoint": cal.get("restore_best_checkpoint"),
            "eval_use_full_holdout": cal.get("eval_use_full_holdout"),
            "preflight": preflight_info,
            "result": payload,
            "status": status,
        }
        summary_path = variant_dir / f"summary_{tag}_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (variant_dir / "latest.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        results.append(summary)

    accs: dict[str, float] = {}
    for s in results:
        if s.get("status") == "ok":
            ev = (s.get("result") or {}).get("eval_official") or (s.get("result") or {}).get("eval") or {}
            acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
            if acc is not None:
                accs[s["variant"]] = float(acc)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "passkey_copy_combined",
        "dry_run": dry_run,
        "official_accuracy": accs,
        "restore_best_checkpoint": cal.get("restore_best_checkpoint"),
        "eval_use_full_holdout": cal.get("eval_use_full_holdout"),
        "errors": errors,
        "results": results,
    }
    combined_path = output_root / f"combined_{tag}_{stamp}.json"
    combined_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    (output_root / "combined_latest.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== Combined summary ===")
    for v in variants:
        a = accs.get(v)
        print(f"  {v}: eval={a * 100:.2f}%" if a is not None else f"  {v}: (missing)")
    print(f"  wrote: {combined_path}\n")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="passkey_copy @ T=4096 — dense/linear training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--variants",
        type=str,
        default="dense_flash",
        help="Comma-separated: dense_flash, linear (default: dense_flash)",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    arena_cfg = load_routing_arena_config(args.config)
    variants = _parse_variants(args.variants, arena_cfg)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors = run_suite(
        dry_run=args.dry_run,
        skip_verify=args.skip_verify,
        variants=variants,
        config_path=args.config,
        output_root=args.output,
    )
    sys.exit(min(errors, 1))


if __name__ == "__main__":
    main()
