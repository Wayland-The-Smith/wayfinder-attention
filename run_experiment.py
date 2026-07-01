#!/usr/bin/env python3
"""
Main entry point for RoutingAttention experiments.

Usage:
  python run_experiment.py --experiment 1 --dry-run
  python run_experiment.py --experiment 1 --variant k64
  python run_experiment.py --experiment 3 --dry-run
  python run_experiment.py --experiment 4 --variant routing_k32
  python run_experiment.py --experiment 5 --variant faiss_hnsw
  python run_experiment.py --list-variants 1

Output structure:
  Experiments/Experiment_N/run_XXX/
    config.yaml
    run_metadata.json
    experiment.log
    checkpoints/
    tensorboard/
    plots/
    stats/
    data_cache/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing_attention.utils.config import load_config, merge_configs

EXPERIMENT_MODULES = {
    1: "experiments.experiment_1",
    2: "experiments.experiment_2",
    3: "experiments.experiment_3",
    4: "experiments.experiment_4",
    5: "experiments.experiment_5",
}


def list_variants(experiment_num: int) -> None:
    cfg_path = ROOT / "configs" / f"experiment_{experiment_num}.yaml"
    if not cfg_path.exists():
        print(f"No config for experiment {experiment_num}")
        return
    cfg = load_config(cfg_path)
    variants = cfg.get("variants", {})
    print(f"Experiment {experiment_num} variants:")
    for name in variants:
        print(f"  - {name}")
    if not variants:
        print("  (no variants defined — uses default config)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RoutingAttention experiments")
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Experiment number (1-5)",
    )
    parser.add_argument(
        "--variant", "-v",
        type=str,
        default=None,
        help="Config variant name (see --list-variants)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline with ~40 steps for sanity checks",
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List available variants for the experiment and exit",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset (default: mnist; legacy: synthetic_mixed)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override sequence length",
    )
    args = parser.parse_args()

    if args.list_variants:
        list_variants(args.experiment)
        return 0

    import importlib
    module_name = EXPERIMENT_MODULES[args.experiment]
    module = importlib.import_module(module_name)

    config_override = {}
    if args.dataset or args.seq_len:
        config_override["data"] = {}
        if args.dataset:
            config_override["data"]["dataset"] = args.dataset
        if args.seq_len:
            config_override["data"]["seq_len"] = args.seq_len
            config_override.setdefault("model", {})["max_seq_len"] = args.seq_len

    print(f"Running Experiment {args.experiment}" +
          (f" variant={args.variant}" if args.variant else "") +
          (" [DRY RUN]" if args.dry_run else ""))

    summary = module.run(
        variant=args.variant,
        dry_run=args.dry_run,
        config_override=config_override if config_override else None,
    )

    print("\n=== Run Complete ===")
    print(f"Run directory: {summary.get('run_dir', 'see Experiments/Experiment_N/run_XXX')}")
    if "verdict" in summary:
        print(f"Verdict: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
