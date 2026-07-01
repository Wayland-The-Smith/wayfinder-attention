#!/usr/bin/env python3
"""
Run the Experiment 4 task-quality gate suite sequentially.

Top-3 retrieval modes from Experiment 1 (Recall@32) plus dense baseline:
  1. routing_asymmetric  — RouterMLP, asymmetric q·k routing (run_052)
  2. learned_address_k32 — decoupled address book (run_051)
  3. key_vector_k32      — dense Q/K projections, no router training
  4. dense               — full attention control

Gate: LM loss and digit accuracy vs dense (not Recall@K).

All variants share the same transformer init (Experiment_1/run_025) and equal
fine-tune steps (fair_finetune).

Usage:
  python run_experiment_4_suite.py
  python run_experiment_4_suite.py --dry-run
  python run_experiment_4_suite.py --start 2 --end 4
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments import experiment_4
from experiments.common import resolve_reuse_transformer_checkpoint
from routing_attention.utils.config import load_config, merge_configs
from routing_attention.utils.experiment import get_experiments_root, resolve_checkpoint_path

TRANSFORMER_CKPT_REL = "Experiment_1/run_025/checkpoints/transformer/final.pt"
ROUTER_ASYMMETRIC_CKPT_REL = "Experiment_1/run_052/checkpoints/router/best.pt"
LEARNED_ADDRESS_CKPT_REL = "Experiment_1/run_051/checkpoints/addresses/best.pt"

SUITE_RUNS: list[dict] = [
    {
        "suite_index": 1,
        "variant": "routing_asymmetric",
        "hypothesis": "Asymmetric RouterMLP (60.6% Recall@32) preserves task quality under sparse attention.",
        "changes": "attention_type=routing, router.similarity=asymmetric, top_k=32, freeze_router",
        "why": "Best Exp-1 recall — primary sparse-attention candidate for task gate.",
        "checkpoint": "Experiment_1/run_052/checkpoints/router/best.pt",
    },
    {
        "suite_index": 2,
        "variant": "learned_address_k32",
        "hypothesis": "Learned address routing (57.2% Recall@32) matches or beats RouterMLP on LM/digit metrics.",
        "changes": "attention_type=learned_address, address_dim=32, freeze_addresses",
        "why": "Second-best Exp-1 recall — tests decoupled search index on real task loss.",
        "checkpoint": "Experiment_1/run_051/checkpoints/addresses/best.pt",
    },
    {
        "suite_index": 3,
        "variant": "key_vector_k32",
        "hypothesis": "Key-vector sparse attention (52.4% Recall@32, no router) is competitive on task metrics.",
        "changes": "attention_type=key_vector, top_k=32, no router checkpoint",
        "why": "Cheap baseline — if it passes task gate, RouterMLP may be unnecessary.",
        "checkpoint": None,
    },
    {
        "suite_index": 4,
        "variant": "dense",
        "hypothesis": "Dense attention is the task-quality ceiling all sparse variants must stay near.",
        "changes": "attention_type=dense, full attention",
        "why": "Control — LM loss / digit accuracy reference for the gate.",
        "checkpoint": None,
    },
]


def _build_config_override(run: dict, dry_run: bool) -> dict:
    return {
        "idea_manifest": {
            "suite_name": "experiment_4_task_gate",
            "suite_index": run["suite_index"],
            "variant": run["variant"],
            "hypothesis": run["hypothesis"],
            "changes": run["changes"],
            "why": run["why"],
            "transformer_checkpoint": TRANSFORMER_CKPT_REL,
            "routing_checkpoint": run.get("checkpoint"),
            "dry_run": dry_run,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
        "reuse": {
            "transformer_checkpoint": TRANSFORMER_CKPT_REL,
        },
    }


def _preflight() -> list[str]:
    warnings: list[str] = []
    base = load_config(ROOT / "configs" / "base.yaml")
    exp_cfg = load_config(ROOT / "configs" / "experiment_4.yaml")
    config = merge_configs(base, exp_cfg)

    if resolve_reuse_transformer_checkpoint(config) is None:
        warnings.append(
            f"Missing transformer checkpoint ({TRANSFORMER_CKPT_REL}) — "
            "Phase comparisons will not be controlled."
        )
    if resolve_checkpoint_path(ROUTER_ASYMMETRIC_CKPT_REL) is None:
        warnings.append(f"Missing asymmetric router checkpoint: {ROUTER_ASYMMETRIC_CKPT_REL}")
    if resolve_checkpoint_path(LEARNED_ADDRESS_CKPT_REL) is None:
        warnings.append(f"Missing learned-address checkpoint: {LEARNED_ADDRESS_CKPT_REL}")
    return warnings


def _task_gate_summary(results: list[dict], max_loss_delta: float) -> dict:
    """Compare sparse variants against dense baseline from suite results."""
    by_variant = {r["variant"]: r for r in results if r.get("status") == "ok"}
    dense = by_variant.get("dense", {})
    dense_loss = dense.get("lm_loss")
    dense_digit = dense.get("digit_accuracy")

    comparisons: list[dict] = []
    for run in SUITE_RUNS:
        variant = run["variant"]
        if variant == "dense":
            continue
        entry = by_variant.get(variant)
        if not entry:
            comparisons.append({"variant": variant, "status": "missing"})
            continue
        loss_delta = None
        digit_delta = None
        gate = "pending_dense"
        if dense_loss is not None and entry.get("lm_loss") is not None:
            loss_delta = entry["lm_loss"] - dense_loss
            gate = "pass" if loss_delta <= max_loss_delta else "fail"
        if dense_digit is not None and entry.get("digit_accuracy") is not None:
            digit_delta = entry["digit_accuracy"] - dense_digit
        comparisons.append({
            "variant": variant,
            "lm_loss": entry.get("lm_loss"),
            "digit_accuracy": entry.get("digit_accuracy"),
            "loss_delta_vs_dense": loss_delta,
            "digit_delta_vs_dense": digit_delta,
            "task_gate": gate,
        })

    return {
        "dense_baseline": {
            "lm_loss": dense_loss,
            "digit_accuracy": dense_digit,
            "run_dir": dense.get("run_dir"),
        },
        "sparse_comparisons": comparisons,
        "max_allowed_loss_delta": max_loss_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 4 task-quality gate suite")
    parser.add_argument("--dry-run", action="store_true", help="40-step smoke test per variant")
    parser.add_argument("--start", type=int, default=1, help="First suite_index (inclusive)")
    parser.add_argument("--end", type=int, default=len(SUITE_RUNS), help="Last suite_index (inclusive)")
    args = parser.parse_args()

    runs = [r for r in SUITE_RUNS if args.start <= r["suite_index"] <= args.end]
    if not runs:
        print(f"No runs in range {args.start}–{args.end}")
        return 1

    suite_log_dir = get_experiments_root() / "Experiment_4" / "suite_task_gate"
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_log_path = suite_log_dir / f"suite_run_{suite_stamp}.jsonl"

    print(f"Experiment 4 task gate suite: {len(runs)} runs")
    print(f"Suite log: {suite_log_path}")
    if args.dry_run:
        print("DRY RUN — 40 fine-tune steps per variant")

    for w in _preflight():
        print(f"WARNING: {w}")

    results: list[dict] = []
    for run in runs:
        label = f"[{run['suite_index']}/{len(SUITE_RUNS)}] {run['variant']}"
        print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}")
        entry = {
            "suite_index": run["suite_index"],
            "variant": run["variant"],
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            summary = experiment_4.run(
                variant=run["variant"],
                dry_run=args.dry_run,
                config_override=_build_config_override(run, args.dry_run),
                compare_all=False,
            )
            entry.update({
                "status": "ok",
                "run_dir": summary.get("run_dir"),
                "verdict": summary.get("verdict"),
                "lm_loss": summary.get("lm_loss"),
                "perplexity": summary.get("perplexity"),
                "digit_accuracy": summary.get("digit_accuracy"),
                "loss_delta_vs_dense": summary.get("loss_delta_vs_dense"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            print(
                f"OK {label} → lm_loss={entry.get('lm_loss')}, "
                f"digit_acc={entry.get('digit_accuracy')}, verdict={entry.get('verdict')}"
            )
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__}"
            entry.update({
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"FAILED {label}: {error_msg}")
            fail_dir = suite_log_dir / f"failed_{run['variant']}_{suite_stamp}"
            fail_dir.mkdir(parents=True, exist_ok=True)
            fail_txt = fail_dir / "failure.txt"
            fail_txt.write_text(
                f"variant: {run['variant']}\n"
                f"suite_index: {run['suite_index']}\n"
                f"hypothesis: {run['hypothesis']}\n"
                f"changes: {run['changes']}\n"
                f"why: {run['why']}\n\n"
                f"error: {error_msg}\n\n"
                f"{traceback.format_exc()}",
                encoding="utf-8",
            )
            entry["failure_log"] = str(fail_txt)

        results.append(entry)
        with open(suite_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    base = load_config(ROOT / "configs" / "base.yaml")
    exp_cfg = load_config(ROOT / "configs" / "experiment_4.yaml")
    config = merge_configs(base, exp_cfg)
    max_loss_delta = config.get("success_criteria", {}).get("lm_loss_increase_max", 0.05)
    task_gate = _task_gate_summary(results, max_loss_delta)

    summary_path = suite_log_dir / f"suite_summary_{suite_stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "suite": "experiment_4_task_gate",
            "dry_run": args.dry_run,
            "runs": results,
            "task_gate": task_gate,
        }, f, indent=2, default=str)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nSuite complete: {ok} ok, {failed} failed")
    print(f"Summary: {summary_path}")

    if task_gate.get("dense_baseline", {}).get("lm_loss") is not None:
        print("\nTask gate (LM loss delta vs dense, max +{:.3f}):".format(max_loss_delta))
        for comp in task_gate.get("sparse_comparisons", []):
            if comp.get("status") == "missing":
                print(f"  {comp['variant']}: MISSING")
                continue
            delta = comp.get("loss_delta_vs_dense")
            gate = comp.get("task_gate", "?")
            digit = comp.get("digit_accuracy")
            print(f"  {comp['variant']}: Δloss={delta:+.4f} digit_acc={digit} → {gate}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
