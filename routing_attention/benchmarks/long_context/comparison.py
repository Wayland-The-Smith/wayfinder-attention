"""Cross-variant comparison plots and tables for Experiment 7."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Display order for head-to-head charts
DEFAULT_VARIANT_ORDER = [
    "dense_flash",
    "routing_asymmetric",
    "learned_address_k32",
    "key_vector_k32",
    "local_window256",
    "local_window64",
    "linear",
]


def _variant_summaries(suite_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run in suite_runs:
        if run.get("status") not in ("ok", "oom_or_eval_error"):
            continue
        var = run.get("variant")
        summary = run.get("summary")
        if var and summary:
            out[var] = summary
    return out


def _ordered_variants(summaries: dict[str, dict[str, Any]]) -> list[str]:
    known = [v for v in DEFAULT_VARIANT_ORDER if v in summaries]
    rest = sorted(v for v in summaries if v not in known)
    return known + rest


def build_comparison_table(suite_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flat table rows for markdown / JSON export."""
    rows: list[dict[str, Any]] = []
    for run in suite_runs:
        if run.get("status") not in ("ok", "oom_or_eval_error"):
            continue
        summary = run.get("summary", {})
        rows.append(
            {
                "variant": run.get("variant"),
                "overall_accuracy": summary.get("overall_accuracy"),
                "by_context_length": summary.get("by_context_length", {}),
                "by_task_type": summary.get("by_task_type", {}),
                "eval_errors": run.get("eval_errors", 0),
                "peak_vram_mb": run.get("peak_vram_mb"),
                "eval_latency_ms": run.get("eval_latency_ms"),
                "tokens_per_sec": run.get("tokens_per_sec"),
            }
        )
    return rows


def plot_cross_variant_by_length(
    suite_runs: list[dict[str, Any]],
    output_path: Path,
    title: str = "Exact Match vs Context Length (all variants)",
) -> None:
    summaries = _variant_summaries(suite_runs)
    if not summaries:
        return

    all_lengths = sorted(
        {int(k) for s in summaries.values() for k in s.get("by_context_length", {})}
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    for var in _ordered_variants(summaries):
        by_len = summaries[var].get("by_context_length", {})
        ys = [by_len.get(str(L), by_len.get(L, np.nan)) for L in all_lengths]
        ax.plot(all_lengths, ys, marker="o", label=var)
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cross_variant_overall_bar(
    suite_runs: list[dict[str, Any]],
    output_path: Path,
    title: str = "Overall Exact Match by Variant",
) -> None:
    summaries = _variant_summaries(suite_runs)
    variants = _ordered_variants(summaries)
    if not variants:
        return
    values = [summaries[v].get("overall_accuracy", 0.0) for v in variants]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(variants, values, color="steelblue", alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cross_variant_by_task(
    suite_runs: list[dict[str, Any]],
    output_path: Path,
    title: str = "Exact Match by Task Type (all variants)",
) -> None:
    summaries = _variant_summaries(suite_runs)
    variants = _ordered_variants(summaries)
    if not variants:
        return
    tasks = sorted({t for s in summaries.values() for t in s.get("by_task_type", {})})
    x = np.arange(len(tasks))
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, var in enumerate(variants):
        by_task = summaries[var].get("by_task_type", {})
        ys = [by_task.get(t, 0.0) for t in tasks]
        ax.bar(x + i * width, ys, width, label=var)
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels(tasks, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_cross_variant_runtime(
    suite_runs: list[dict[str, Any]],
    output_path: Path,
) -> None:
    variants = []
    latency = []
    vram = []
    for run in suite_runs:
        if run.get("status") not in ("ok", "oom_or_eval_error"):
            continue
        if run.get("eval_latency_ms") is None and run.get("peak_vram_mb") is None:
            continue
        variants.append(run.get("variant", "?"))
        latency.append(run.get("eval_latency_ms") or 0.0)
        vram.append(run.get("peak_vram_mb") or 0.0)
    if not variants:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(variants, latency, color="coral", alpha=0.85)
    ax1.set_ylabel("Eval Forward Latency (ms)")
    ax1.set_title("Latency at benchmark T")
    ax1.tick_params(axis="x", rotation=35)

    ax2.bar(variants, vram, color="seagreen", alpha=0.85)
    ax2.set_ylabel("Peak VRAM (MB)")
    ax2.set_title("Peak GPU Memory During Eval")
    ax2.tick_params(axis="x", rotation=35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_comparison_plots(suite_runs: list[dict[str, Any]], output_dir: Path) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    specs = [
        ("cross_variant_by_length.png", plot_cross_variant_by_length),
        ("cross_variant_overall.png", plot_cross_variant_overall_bar),
        ("cross_variant_by_task.png", plot_cross_variant_by_task),
        ("cross_variant_runtime.png", plot_cross_variant_runtime),
    ]
    for name, fn in specs:
        path = output_dir / name
        fn(suite_runs, path)
        saved.append(str(path))
    return saved
