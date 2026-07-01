#!/usr/bin/env python3
"""
Dense vs linear — addr_val + 2 decoys, bunched @ T=512, 6L, final-checkpoint eval.

Replicates gap_calibration P2_decoys_2 (+23pp @ 3k) with:
  - restore_best_checkpoint: false  (official eval = last training step weights)
  - eval_use_full_holdout: true    (300-sample official holdout)
  - 40k steps (full run)

Usage:
  python scripts/verify_gap_decoys2_bunched.py
  python run_gap_decoys2_final_ckpt_suite.py --dry-run
  python run_gap_decoys2_final_ckpt_suite.py
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

CONFIG_PATH = ROOT / "configs" / "routing_gap_decoys2_bunched_t512_6L_40k_final_ckpt.yaml"
OUTPUT_ROOT = ROOT / "experiments" / "Experiment_7" / "gap_decoys2_bunched_t512_6L_40k_final_ckpt"
VARIANTS = ("dense_flash", "linear")
VERIFY_SCRIPT = ROOT / "scripts" / "verify_gap_decoys2_bunched.py"

DENSE_TARGET_MIN = 0.30
GAP_TARGET_MIN = 0.15


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== gap decoys2 final-ckpt preflight ===")
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


def run_suite(*, dry_run: bool, skip_verify: bool) -> int:
    if not skip_verify:
        run_verify()

    arena_cfg = load_routing_arena_config(CONFIG_PATH)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = int(arena_cfg.get("n_layers", 6))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    preflight_info = preflight(list(VARIANTS), dry_run)
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    bench = _resolve_synthetic_bench_cfg(config, train_t)
    cal = config.get("dense_calibration", {})
    transformer_cfg = config.get("transformer", {})
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("dense_pretrain_steps")
        or 0
    )

    print("=== Experiment plan ===")
    print(f"  task=addr_val  T={train_t}  scatter={bench.scatter_multi_needles}")
    print(f"  decoys={bench.num_distractors}  layers={n_layers}  label={bench.train_label_mode}")
    print(f"  steps={steps}  variants={list(VARIANTS)}  dry_run={dry_run}")
    print(f"  restore_best_checkpoint={cal.get('restore_best_checkpoint')}")
    print(f"  eval_use_full_holdout={cal.get('eval_use_full_holdout')}")
    print(f"  output={OUTPUT_ROOT}")
    if not dry_run:
        print("  estimated wall time: ~25–35 min (2 variants × 40k @ T=512)")
    print()

    results: list[dict] = []
    errors = 0
    for variant in VARIANTS:
        print(f"\n########## decoys_2_final_ckpt / {variant} ##########")
        variant_dir = OUTPUT_ROOT / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
        log_path = variant_dir / log_name
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
        print(f"  log: {log_path}")

        device = init_arena_runtime(config)
        log = logging.getLogger(f"gap_decoys2_final_ckpt.{variant}")
        reset_peak_vram(device)
        try:
            payload = _run_variant(variant, config=config, train_t=train_t, device=device, log=log)
            ev = payload.get("eval_official") or payload.get("eval", {})
            acc = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
            restored = (payload.get("train_info") or {}).get("restored_best_checkpoint")
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
            "kind": "gap_decoys2_final_ckpt",
            "variant": variant,
            "dry_run": dry_run,
            "train_context_length": train_t,
            "n_layers": n_layers,
            "num_distractors": bench.num_distractors,
            "scatter_multi_needles": bench.scatter_multi_needles,
            "training_steps": steps,
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

    gap_pp = None
    if "dense_flash" in accs and "linear" in accs:
        gap_pp = accs["dense_flash"] - accs["linear"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "gap_decoys2_final_ckpt_combined",
        "dry_run": dry_run,
        "official_accuracy": accs,
        "dense_minus_linear_pp": gap_pp,
        "restore_best_checkpoint": cal.get("restore_best_checkpoint"),
        "eval_use_full_holdout": cal.get("eval_use_full_holdout"),
        "success_criteria": {
            "dense_min": DENSE_TARGET_MIN,
            "gap_min_pp": GAP_TARGET_MIN,
            "dense_met": accs.get("dense_flash", 0) >= DENSE_TARGET_MIN if accs else False,
            "gap_met": gap_pp is not None and gap_pp >= GAP_TARGET_MIN,
        },
        "errors": errors,
        "results": results,
    }
    combined_path = OUTPUT_ROOT / f"combined_{tag}_{stamp}.json"
    combined_path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    (OUTPUT_ROOT / "combined_latest.json").write_text(
        json.dumps(combined, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== Combined summary ===")
    for v in VARIANTS:
        a = accs.get(v)
        print(f"  {v}: official={a * 100:.2f}%" if a is not None else f"  {v}: (missing)")
    if gap_pp is not None:
        print(f"  dense - linear: {gap_pp * 100:.2f} pp")
    print(f"  restore_best_checkpoint: {cal.get('restore_best_checkpoint')}")
    print(f"  wrote: {combined_path}\n")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="addr_val decoys_2 bunched @ T=512 — dense vs linear, final checkpoint eval"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    errors = run_suite(dry_run=args.dry_run, skip_verify=args.skip_verify)
    sys.exit(min(errors, 1))


if __name__ == "__main__":
    main()
