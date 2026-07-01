#!/usr/bin/env python3
"""32-sample dense overfit diagnostic for ptr_chain (1 hop, 0 distractors)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context.feasibility_ladder import run_dense_level
from routing_attention.benchmarks.long_context.runtime import collect_device_info
from routing_attention.models.fast_attention import backend_status

import torch


def load_diagnostic_config(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return raw.get("dense_overfit_diagnostic", raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense ptr_chain 32-sample overfit diagnostic")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dense_overfit_ptr_chain_diagnostic.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "Experiment_7" / "dense_overfit_ptr_chain_diagnostic",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    diag_cfg = load_diagnostic_config(args.config)
    level = dict(diag_cfg["level"])
    ladder_cfg = {
        "suite_profile": diag_cfg.get("suite_profile", "full"),
        "synthetic": diag_cfg.get("synthetic", True),
        "holdout": diag_cfg.get("holdout", {}),
        "dry_run": diag_cfg.get("dry_run", {}),
        "levels": [level],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = collect_device_info(device)
    device_info.update(backend_status())
    device_info["dry_run"] = args.dry_run

    print("=== Dense overfit diagnostic preflight ===")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = run_dense_level(
        ladder_cfg,
        level,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
        device_info=device_info,
        dense_checkpoint=None,
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = args.output_dir / f"overfit_diagnostic_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    latest = args.output_dir / "latest.json"
    latest.write_text(json.dumps(result, indent=2), encoding="utf-8")

    passed = result.get("pass_result", {}).get("passed", False)
    acc = result.get("final_eval", {}).get("primary_gate_accuracy")
    print("\n=== Overfit diagnostic summary ===")
    print(f"  passed={passed}")
    if acc is not None:
        print(f"  accuracy={float(acc):.2%} ({result['final_eval'].get('correct', '?')}/{result['final_eval'].get('total', '?')})")
    print(f"  wrote: {out_path}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
