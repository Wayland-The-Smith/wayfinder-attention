"""Markdown report generation for Experiment 7 suite results."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from routing_attention.benchmarks.long_context.comparison import build_comparison_table
from routing_attention.benchmarks.long_context.success_criteria import evaluate_success_criteria


def _pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{x:.1f}"


def generate_markdown_report(
    suite_summary: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> str:
    """Build a markdown report from a suite summary dict (``runs`` list required)."""
    runs = suite_summary.get("runs", [])
    criteria_cfg = suite_summary.get("success_criteria", {})
    variant_results = {
        r["variant"]: r for r in runs if r.get("variant") and r.get("summary")
    }
    criteria = evaluate_success_criteria(variant_results, criteria_cfg)

    lines = [
        "# Experiment 7 — Long-Context Retrieval Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Dry run: `{suite_summary.get('dry_run', False)}`",
        "",
        "## Preflight",
        "",
    ]
    preflight = suite_summary.get("preflight", {})
    for k, v in preflight.items():
        lines.append(f"- **{k}**: {v}")

    lines.extend(
        [
            "",
            "## Suite Status",
            "",
            f"- Variants OK: **{suite_summary.get('variants_ok', 0)}** / "
            f"{suite_summary.get('variants_total', 0)}",
            f"- Hard errors: **{suite_summary.get('variants_error', 0)}**",
            f"- OOM / eval errors: **{suite_summary.get('variants_oom_or_eval_error', 0)}**",
            "",
            "## Success Criteria",
            "",
            f"**Tier:** `{criteria['tier']}`",
            "",
        ]
    )
    for name, check in criteria["checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        lines.append(f"- **{name}**: {status}")
        for k, v in check.items():
            if k == "pass":
                continue
            lines.append(f"  - {k}: {v}")
    lines.append("")

    lines.extend(["## Overall Accuracy by Variant", "", "| Variant | Overall | T≥8k avg | Errors | VRAM MB | Latency ms |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    table = build_comparison_table(runs)
    for row in table:
        by_len = row.get("by_context_length", {})
        long_vals = [float(v) for k, v in by_len.items() if int(k) >= 8192]
        long_avg = sum(long_vals) / len(long_vals) if long_vals else None
        lines.append(
            f"| {row['variant']} | {_pct(row.get('overall_accuracy'))} | "
            f"{_pct(long_avg)} | {row.get('eval_errors', 0)} | "
            f"{_fmt_ms(row.get('peak_vram_mb'))} | {_fmt_ms(row.get('eval_latency_ms'))} |"
        )
    lines.append("")

    lines.extend(["## Accuracy by Task Type", ""])
    variants = [r["variant"] for r in table]
    tasks = sorted({t for r in table for t in r.get("by_task_type", {})})
    if tasks and variants:
        header = "| Task | " + " | ".join(variants) + " |"
        sep = "| --- | " + " | ".join(["---:"] * len(variants)) + " |"
        lines.extend([header, sep])
        for task in tasks:
            cells = []
            for r in table:
                acc = r.get("by_task_type", {}).get(task)
                cells.append(_pct(acc) if acc is not None else "n/a")
            lines.append(f"| {task} | " + " | ".join(cells) + " |")
        lines.append("")

    failures: list[dict[str, Any]] = []
    for run in runs:
        failures.extend(run.get("top_failures", [])[:5])
    if failures:
        lines.extend(["## Sample Failures", ""])
        for f in failures[:15]:
            lines.append(
                f"- **{f.get('task_type')}** T={f.get('context_length')} "
                f"d={f.get('needle_depth')}: expected `{f.get('expected')}` "
                f"got `{f.get('predicted')}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- **dense_flash** is the fair dense baseline at long context; naive **dense** may OOM at 32k.",
            "- Latency is a single eval forward at the configured benchmark context length.",
            "- For full scaling latency vs Flash, see Experiment 6 results.",
            "",
            "## Recommendations",
            "",
        ]
    )
    if criteria["tier"] in ("none", "interesting"):
        lines.append(
            "- Routing has not yet met strong/breakthrough criteria — consider more training steps "
            "or router fine-tuning on this benchmark before Exp 8."
        )
    else:
        lines.append(
            "- Strong tier met — proceed to TinyStories LM eval (Exp 8) and unified comparison table."
        )
    if suite_summary.get("variants_error", 0) > 0:
        lines.append("- Fix hard-error variants before publishing results.")
    lines.append("")

    text = "\n".join(lines)
    if output_path is not None:
        Path(output_path).write_text(text, encoding="utf-8")
    return text
