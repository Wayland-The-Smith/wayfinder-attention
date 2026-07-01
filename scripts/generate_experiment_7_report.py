#!/usr/bin/env python3
"""Regenerate Experiment 7 markdown report and comparison plots from a suite summary JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from routing_attention.benchmarks.long_context import (
    evaluate_success_criteria,
    generate_markdown_report,
    save_comparison_plots,
)
from routing_attention.benchmarks.long_context.comparison import build_comparison_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp 7 report from suite summary JSON")
    parser.add_argument("summary_json", type=Path, help="Path to suite_*_summary_*.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    out_dir = args.output_dir or args.summary_json.parent

    criteria_cfg = summary.get("success_criteria", {})
    variant_results = {r["variant"]: r for r in summary.get("runs", []) if r.get("summary")}
    summary["success_evaluation"] = evaluate_success_criteria(variant_results, criteria_cfg)
    summary["comparison_table"] = build_comparison_table(summary.get("runs", []))

    tag = "dry_run" if summary.get("dry_run") else "full"
    plots_dir = out_dir / f"comparison_regen_{tag}"
    summary["comparison_plots"] = save_comparison_plots(summary.get("runs", []), plots_dir)

    report_path = out_dir / f"REPORT_regen_{args.summary_json.stem}.md"
    generate_markdown_report(summary, output_path=report_path)
    print(f"Report: {report_path}")
    print(f"Plots: {plots_dir}")
    print(f"Success tier: {summary['success_evaluation']['tier']}")


if __name__ == "__main__":
    main()
