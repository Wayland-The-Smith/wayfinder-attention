#!/usr/bin/env python3
"""
NIAH feasibility ladder — sequential Level 0–4 gates before variant breakthrough.

Level 0: stack smoke (T=512, suffix-after-needle, L0 overfit 32)
Level 1: distance isolated (L1a — T=2048, pointer_unique, suffix after needles)
Level 2: distance curriculum (L1′ — suffix near-needle → full random)
Level 3: full gate (T=8192, L0–L4, context curriculum) → saves dense checkpoint
Level 4: variant comparison (dense / linear / local / routing) at T=8192

Usage:
  python run_feasibility_ladder_suite.py --dry-run
  python run_feasibility_ladder_suite.py
  python run_feasibility_ladder_suite.py --levels 0 1
  python run_feasibility_ladder_suite.py --dry-run --max-steps 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.feasibility_ladder import (
    load_feasibility_ladder_config,
    run_feasibility_ladder,
)
from routing_attention.benchmarks.long_context.runtime import collect_device_info
from routing_attention.models.fast_attention import backend_status

import torch


def preflight(dry_run: bool) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = collect_device_info(device)
    info.update(backend_status())
    info["dry_run"] = dry_run
    print("=== Feasibility ladder preflight ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if info["device_type"] != "cuda":
        print("WARNING: CUDA not available — runs will be slow and not representative.")
    if not info.get("fla_linear"):
        print("ERROR: flash-linear-attention required.")
        sys.exit(1)
    print()
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="NIAH feasibility ladder (Levels 0–4)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run (~50 steps/level) to verify train+eval pipeline",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=None,
        help="Run only these levels (default: all 0–4)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "feasibility_ladder.yaml",
        help="Ladder config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory (default: experiments/Experiment_7/feasibility_ladder)",
    )
    parser.add_argument(
        "--no-stop-on-failure",
        action="store_true",
        help="Continue to next level even if pass criteria fail",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override dry-run max steps (only with --dry-run)",
    )
    args = parser.parse_args()

    preflight(args.dry_run)

    ladder_cfg = load_feasibility_ladder_config(args.config)
    if args.dry_run and args.max_steps is not None:
        ladder_cfg.setdefault("dry_run", {})["max_steps"] = args.max_steps
        ladder_cfg["dry_run"]["validate_every"] = max(5, args.max_steps // 5)
        ladder_cfg["dry_run"]["log_every"] = max(5, args.max_steps // 5)

    print("=== Feasibility ladder plan ===")
    print(f"  profile: {ladder_cfg.get('suite_profile', 'full')}")
    print(f"  synthetic: {ladder_cfg.get('synthetic', True)}")
    dry = ladder_cfg.get("dry_run", {})
    if args.dry_run:
        print(f"  dry_run steps: {dry.get('max_steps', 50)}")
    for lv in ladder_cfg.get("levels", []):
        if args.levels is None or int(lv["level"]) in args.levels:
            print(
                f"  Level {lv['level']}: {lv.get('name')} "
                f"T={lv.get('train_context_length')} "
                f"steps={lv.get('transformer', {}).get('max_steps', '?')}"
            )
    print()

    if args.config != ROOT / "configs" / "feasibility_ladder.yaml":
        import yaml

        out_cfg = args.output_dir or (ROOT / "experiments" / "Experiment_7" / "feasibility_ladder")
        out_cfg.mkdir(parents=True, exist_ok=True)
        (out_cfg / "ladder_config_used.yaml").write_text(
            yaml.safe_dump({"feasibility_ladder": ladder_cfg}, sort_keys=False),
            encoding="utf-8",
        )

    summary = run_feasibility_ladder(
        dry_run=args.dry_run,
        levels=args.levels,
        config_path=args.config,
        output_dir=args.output_dir,
        stop_on_failure=not args.no_stop_on_failure,
    )

    if args.dry_run:
        level_errors = [e for e in summary.get("levels", []) if e.get("status") == "error"]
        variant_errors = [
            (e.get("level"), var, vr.get("error"))
            for e in summary.get("levels", [])
            for var, vr in (e.get("variant_results") or {}).items()
            if vr.get("status") == "error"
        ]
        if level_errors or variant_errors or (
            summary.get("aborted") and any(e.get("status") == "error" for e in summary.get("levels", []))
        ):
            if variant_errors:
                for lid, var, err in variant_errors:
                    print(f"ERROR Level {lid} variant {var}: {err}")
            sys.exit(1)
        print("Dry-run completed — all levels executed train+eval without crash.")
    elif summary.get("aborted"):
        sys.exit(1)


if __name__ == "__main__":
    main()
