#!/usr/bin/env python3
"""
Run the Experiment 1 ablation suite (router ideas) sequentially.

Reuses transformer + attention cache from run_025 by default (see configs/experiment_1.yaml).
Skips Phase A when checkpoint loads; skips Phase B when compatible cache is reused.

Usage:
  python run_experiment_1_ideas.py
  python run_experiment_1_ideas.py --start 3 --end 5
  python run_experiment_1_ideas.py --dry-run
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

from experiments import experiment_1
from experiments.common import resolve_baseline_run_dir, resolve_reuse_transformer_checkpoint
from routing_attention.utils.experiment import get_experiments_root

# Ordered for full-suite runs: most likely to make routing attention work → least likely.
# Tier 1 = core mechanism + control; 2 = training/arch levers; 3 = alt paradigm;
# 4 = deployment diagnostic; 5 = ruled-out / completeness ablations (run last).
SUITE_IDEAS: list[dict] = [
    {
        "suite_index": 1,
        "variant": "learned_address",
        "reuse_cache": True,
        "hypothesis": "Decoupled learned addresses (q_addr/k_addr) beat RouterMLP and key-vector baselines.",
        "changes": "eval_mode=learned_address, per-layer q_addr_proj/k_addr_proj, meat Q/K frozen in Phase C",
        "why": "Target architecture — search index separate from attention meat; train addresses with routing loss.",
        "tier": 1,
    },
    {
        "suite_index": 2,
        "variant": "asymmetric",
        "reuse_cache": True,
        "hypothesis": "Directed Q/K routing (q_i·k_j) beats symmetric r_i·r_j on mid/deep layers.",
        "changes": "router.similarity=asymmetric",
        "why": "Best RouterMLP dry-run (56.1%) — compare against learned_address.",
        "tier": 1,
    },
    {
        "suite_index": 3,
        "variant": "key_vector_baseline",
        "reuse_cache": True,
        "hypothesis": "Dedicated RouterMLP may be unnecessary if Q/K are already searchable.",
        "changes": "skip_router_training, eval_mode=key_vector (dense Q/K projections)",
        "why": "52.4% with no router training — bar asymmetric must beat to justify RouterMLP.",
        "tier": 1,
    },
    {
        "suite_index": 4,
        "variant": "router_8k_steps",
        "reuse_cache": True,
        "hypothesis": "4000 router steps/layer under-trains harder layers.",
        "changes": "router.max_steps=8000",
        "why": "Best symmetric training lever — L2/L5 still moving at 4k in run_025.",
        "tier": 2,
    },
    {
        "suite_index": 5,
        "variant": "train_k64_eval_k32",
        "reuse_cache": True,
        "hypothesis": "Training with top-64 positives helps ranking at eval K=32.",
        "changes": "router.top_k=64, evaluation.recall_k=32",
        "why": "Richer training neighborhoods may improve Recall@32 deployment metric.",
        "tier": 2,
    },
    {
        "suite_index": 6,
        "variant": "dim64",
        "reuse_cache": True,
        "hypothesis": "32-d routing space lacks capacity for harder layers.",
        "changes": "router.routing_dim=64, hidden_dim=128",
        "why": "Capacity sweep — may lift weak layers if routing dim is the bottleneck.",
        "tier": 2,
    },
    {
        "suite_index": 7,
        "variant": "k64",
        "reuse_cache": True,
        "hypothesis": "Recall is capped because K=32 is too strict for MNIST neighborhoods.",
        "changes": "router.top_k=64, evaluation.recall_k=64",
        "why": "Tests whether sparse attention needs wider neighborhoods to be useful.",
        "tier": 2,
    },
    {
        "suite_index": 8,
        "variant": "joint_aux_light",
        "reuse_cache": False,
        "hypothesis": "Joint task+routing aux loss produces more matchable hidden states.",
        "changes": "Phase A2: 2000 steps, routing_aux_weight=0.1, skip separate router training",
        "why": "Co-adaptation path — weak dry-run (25.8%) but only way to test joint training.",
        "tier": 3,
    },
    {
        "suite_index": 9,
        "variant": "eval_last_layer_only",
        "reuse_cache": True,
        "hypothesis": "Mean across layers hides strong early-layer signal (L0≈54%).",
        "changes": "evaluation.eval_layers=[-1] only (still trains all layers)",
        "why": "Tells you which layers to route in a deployed sparse-attention stack.",
        "tier": 4,
    },
    {
        "suite_index": 10,
        "variant": "per_head_supervision",
        "reuse_cache": False,
        "hypothesis": "Head-averaged supervision blurs per-head attention structure.",
        "changes": "data_collection.per_head=true, attention_supervision=per_head (new cache)",
        "why": "Dry-run 30.0% — unlikely on MNIST; slow cache; completeness only.",
        "tier": 5,
    },
    {
        "suite_index": 11,
        "variant": "mse_baseline",
        "reuse_cache": True,
        "hypothesis": "Matrix factorization (MSE) beats neighborhood InfoNCE on MNIST.",
        "changes": "router.loss_type=mse, mode=single, last layer only",
        "why": "Dry-run 16.1% (below random) — ruled out; run last for ablation table.",
        "tier": 5,
    },
]


def _build_config_override(idea: dict, dry_run: bool) -> dict:
    override: dict = {
        "idea_manifest": {
            "suite_name": "experiment_1_ideas",
            "suite_index": idea["suite_index"],
            "variant": idea["variant"],
            "tier": idea["tier"],
            "hypothesis": idea["hypothesis"],
            "changes": idea["changes"],
            "why": idea["why"],
            "reuse_cache": idea.get("reuse_cache", True),
            "baseline_run": "Experiment_1/run_025",
            "dry_run": dry_run,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    if not idea.get("reuse_cache", True):
        override["reuse"] = {
            "attention_cache_train": None,
            "attention_cache_holdout": None,
        }
    return override


def _preflight(config_path: Path) -> list[str]:
    warnings: list[str] = []
    from routing_attention.utils.config import load_config, merge_configs, apply_variant

    base = load_config(ROOT / "configs" / "base.yaml")
    exp_cfg = load_config(config_path)
    config = merge_configs(base, exp_cfg)

    run_dir = resolve_baseline_run_dir(config)
    if run_dir is None:
        warnings.append("baseline run_025 not found — Phase A/B may run from scratch.")
    else:
        ckpt = resolve_reuse_transformer_checkpoint(config)
        if ckpt is None:
            warnings.append(
                f"No transformer checkpoint in {run_dir} — Phase A (10k steps) will run once. "
                "Consider re-running a single `python run_experiment.py --experiment 1` to save final.pt."
            )
        train_cache = run_dir / "data_cache" / "attention_cache_train_all_layers"
        holdout_cache = run_dir / "data_cache" / "attention_cache_holdout_all_layers"
        if not (train_cache / "manifest.json").exists():
            warnings.append(f"Missing train cache at {train_cache}")
        if not (holdout_cache / "manifest.json").exists():
            warnings.append(f"Missing holdout cache at {holdout_cache}")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Experiment 1 ablation suite")
    parser.add_argument("--dry-run", action="store_true", help="40-step smoke test per idea")
    parser.add_argument("--start", type=int, default=1, help="First suite_index (inclusive)")
    parser.add_argument("--end", type=int, default=len(SUITE_IDEAS), help="Last suite_index (inclusive)")
    args = parser.parse_args()

    ideas = [i for i in SUITE_IDEAS if args.start <= i["suite_index"] <= args.end]
    if not ideas:
        print(f"No ideas in range {args.start}–{args.end}")
        return 1

    suite_log_dir = get_experiments_root() / "Experiment_1" / "suite_ideas"
    suite_log_dir.mkdir(parents=True, exist_ok=True)
    suite_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suite_log_path = suite_log_dir / f"suite_run_{suite_stamp}.jsonl"

    print(f"Experiment 1 ideas suite: {len(ideas)} runs")
    print(f"Suite log: {suite_log_path}")

    for w in _preflight(ROOT / "configs" / "experiment_1.yaml"):
        print(f"WARNING: {w}")

    results: list[dict] = []
    for idea in ideas:
        label = f"[{idea['suite_index']}/{len(SUITE_IDEAS)}] {idea['variant']}"
        print(f"\n{'=' * 60}\n>>> {label}\n{'=' * 60}")
        entry = {
            "suite_index": idea["suite_index"],
            "variant": idea["variant"],
            "status": "pending",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            summary = experiment_1.run(
                variant=idea["variant"],
                dry_run=args.dry_run,
                config_override=_build_config_override(idea, args.dry_run),
            )
            entry.update({
                "status": "ok",
                "run_dir": summary.get("run_dir"),
                "verdict": summary.get("verdict"),
                "recall": (summary.get("recall_metrics") or {}).get("mean_recall"),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"OK {label} → verdict={entry.get('verdict')}, recall={entry.get('recall')}")
        except Exception as exc:
            error_msg = str(exc) or f"{type(exc).__name__}"
            entry.update({
                "status": "error",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            })
            print(f"FAILED {label}: {error_msg}")
            fail_dir = get_experiments_root() / "Experiment_1" / "suite_ideas" / f"failed_{idea['variant']}_{suite_stamp}"
            fail_dir.mkdir(parents=True, exist_ok=True)
            fail_txt = fail_dir / "failure.txt"
            fail_txt.write_text(
                f"variant: {idea['variant']}\n"
                f"suite_index: {idea['suite_index']}\n"
                f"hypothesis: {idea['hypothesis']}\n"
                f"changes: {idea['changes']}\n"
                f"why: {idea['why']}\n\n"
                f"error: {error_msg}\n\n"
                f"{traceback.format_exc()}",
                encoding="utf-8",
            )
            entry["failure_log"] = str(fail_txt)

        results.append(entry)
        with open(suite_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    summary_path = suite_log_dir / f"suite_summary_{suite_stamp}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"ideas": results, "dry_run": args.dry_run}, f, indent=2, default=str)

    ok = sum(1 for r in results if r["status"] == "ok")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"\nSuite complete: {ok} ok, {failed} failed")
    print(f"Summary: {summary_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
