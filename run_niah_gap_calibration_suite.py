#!/usr/bin/env python3
"""
Dense vs linear gap probes — pointer_unique NIAH and scattered multi-query MQAR.

Experiments:
  pointer_unique  — scattered needle + decoys @ T=4096 (sharp peak / NIAH)
  mqar_scatter    — mqar_addr_val N=16, Q=8, scatter @ T=2048

Usage:
  python scripts/verify_niah_gap_tasks.py
  python run_niah_gap_calibration_suite.py --experiment pointer_unique --dry-run
  python run_niah_gap_calibration_suite.py --experiment mqar_scatter --dry-run
  python run_niah_gap_calibration_suite.py --experiment all --dry-run
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

EXPERIMENTS: dict[str, Path] = {
    "pointer_unique": ROOT / "configs" / "routing_pointer_unique_niah_t4096.yaml",
    "mqar_scatter": ROOT / "configs" / "routing_mqar_scatter_multiquery_t2048.yaml",
}
VARIANTS = ("dense_flash", "linear")
VERIFY_SCRIPT = ROOT / "scripts" / "verify_niah_gap_tasks.py"


def preflight(variants: list[str], dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    info["variants"] = variants
    print("=== NIAH gap calibration preflight ===")
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


def _default_output_root(experiment: str, arena_cfg: dict) -> Path:
    bench = arena_cfg.get("long_context_benchmark", {})
    label = bench.get("benchmark_variant") or experiment
    return ROOT / "experiments" / "Experiment_7" / f"niah_gap_{label}"


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


def run_experiment(
    experiment: str,
    *,
    config_path: Path,
    output_dir: Path | None,
    dry_run: bool,
    skip_verify: bool,
) -> int:
    if not skip_verify:
        run_verify()

    arena_cfg = load_routing_arena_config(config_path)
    train_t = int(arena_cfg["train_context_length"])
    n_layers = int(arena_cfg.get("n_layers", 4))
    output_root = output_dir or _default_output_root(experiment, arena_cfg)
    output_root.mkdir(parents=True, exist_ok=True)

    preflight_info = preflight(list(VARIANTS), dry_run)
    config = build_arena_experiment_config(arena_cfg, dry_run=dry_run, n_layers=n_layers)
    bench = _resolve_synthetic_bench_cfg(config, train_t)
    transformer_cfg = config.get("transformer", {})
    steps = int(
        transformer_cfg.get("sparse_finetune_steps")
        or transformer_cfg.get("dense_pretrain_steps")
        or 0
    )

    print("=== Experiment plan ===")
    print(f"  experiment={experiment}")
    print(f"  task={bench.task_types[0]}  T={train_t}  scatter={bench.scatter_multi_needles}")
    if bench.task_types == ["mqar_addr_val"]:
        print(f"  N={bench.num_kv_pairs}  queries={bench.num_queries}  label={bench.train_label_mode}")
    else:
        print(f"  decoys={bench.num_distractors}  label={bench.train_label_mode}")
    print(f"  steps={steps}  variants={list(VARIANTS)}  dry_run={dry_run}")
    print(f"  output={output_root}")
    print()

    results: list[dict] = []
    errors = 0
    for variant in VARIANTS:
        print(f"\n########## {experiment} / {variant} ##########")
        variant_dir = output_root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        log_name = f"run_dry_{variant}.log" if dry_run else f"run_{variant}.log"
        log_path = variant_dir / log_name
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(fh)
        print(f"  log: {log_path}")

        device = init_arena_runtime(config)
        log = logging.getLogger(f"niah_gap.{experiment}.{variant}")
        reset_peak_vram(device)
        try:
            payload = _run_variant(variant, config=config, train_t=train_t, device=device, log=log)
            ev = payload.get("eval_official") or payload.get("eval", {})
            acc = float(ev.get("primary_gate_accuracy", ev.get("overall_accuracy", 0)))
            print(
                f"OK {variant}: official_gate={acc * 100:.2f}% "
                f"({ev.get('primary_gate_correct', ev.get('correct'))}/"
                f"{ev.get('primary_gate_total', ev.get('total'))})"
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
            "kind": "niah_gap_calibration",
            "experiment": experiment,
            "variant": variant,
            "dry_run": dry_run,
            "train_context_length": train_t,
            "task_type": bench.task_types[0],
            "scatter_multi_needles": bench.scatter_multi_needles,
            "training_steps": steps,
            "preflight": preflight_info,
            "result": payload,
            "status": status,
        }
        if bench.task_types == ["mqar_addr_val"]:
            summary["num_kv_pairs"] = bench.num_kv_pairs
            summary["num_queries"] = bench.num_queries
        else:
            summary["num_distractors"] = bench.num_distractors
        summary_path = variant_dir / f"summary_{tag}_{stamp}.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        (variant_dir / "latest.json").write_text(
            json.dumps(summary, indent=2, default=str),
            encoding="utf-8",
        )
        results.append(summary)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = "dry_run" if dry_run else "full"
    accs = {}
    for s in results:
        if s.get("status") == "ok":
            ev = (s.get("result") or {}).get("eval_official") or (s.get("result") or {}).get("eval") or {}
            acc = ev.get("primary_gate_accuracy", ev.get("overall_accuracy"))
            if acc is not None:
                accs[s["variant"]] = float(acc)
    combined = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": "niah_gap_calibration_combined",
        "experiment": experiment,
        "dry_run": dry_run,
        "official_accuracy": accs,
        "dense_minus_linear_pp": (
            (accs.get("dense_flash") - accs.get("linear"))
            if "dense_flash" in accs and "linear" in accs
            else None
        ),
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
    for v in VARIANTS:
        a = accs.get(v)
        print(f"  {v}: official={a * 100:.2f}%" if a is not None else f"  {v}: (missing)")
    if combined["dense_minus_linear_pp"] is not None:
        print(f"  dense - linear: {combined['dense_minus_linear_pp'] * 100:.2f} pp")
    print(f"  wrote: {combined_path}\n")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="NIAH / MQAR dense vs linear gap calibration")
    parser.add_argument(
        "--experiment",
        choices=("pointer_unique", "mqar_scatter", "all"),
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path, default=None, help="Override config (single experiment)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.config is not None:
        code = run_experiment(
            "custom",
            config_path=args.config,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            skip_verify=args.skip_verify,
        )
        sys.exit(min(code, 1))

    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    total_errors = 0
    for name in names:
        total_errors += run_experiment(
            name,
            config_path=EXPERIMENTS[name],
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            skip_verify=args.skip_verify and name != names[0],
        )
    sys.exit(min(total_errors, 1))


if __name__ == "__main__":
    main()
