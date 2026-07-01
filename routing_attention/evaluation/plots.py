"""Plotting utilities for experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns


def plot_recall_by_distance(
    per_bin: list[dict[str, Any]],
    output_path: Path,
    title: str = "Recall@K by Token Distance",
) -> None:
    """Plot recall vs distance bin for Experiment 2."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"({b['distance_min']}, {b['distance_max']}]" for b in per_bin]
    recalls = [b["recall"] for b in per_bin]
    totals = [b["total"] for b in per_bin]

    bars = ax.bar(range(len(labels)), recalls, color="steelblue", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Token Distance |i - j|")
    ax.set_ylabel("Recall@K")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)

    for bar, total in zip(bars, totals):
        if total == 0:
            bar.set_alpha(0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_comparison_bar(
    metrics: dict[str, float],
    output_path: Path,
    title: str = "Model Comparison",
    ylabel: str = "Value",
) -> None:
    """Bar chart comparing models on a single metric."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(metrics.keys())
    values = list(metrics.values())
    ax.bar(names, values, color="coral", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curve(
    history: dict[str, list[dict[str, Any]]],
    tag: str,
    output_path: Path,
    title: str | None = None,
) -> None:
    if tag not in history:
        return
    steps = [e["step"] for e in history[tag]]
    values = [e["value"] for e in history[tag]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, values, marker="o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel(tag)
    ax.set_title(title or tag)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_scale_curves(
    seq_lengths: list[int],
    metrics_by_seq: dict[str, list[float]],
    output_path: Path,
    title: str = "Scaling Behavior",
    ylabel: str = "Metric",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, values in metrics_by_seq.items():
        ax.plot(seq_lengths, values, marker="o", label=name)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xscale("log", base=2)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_all_plots(
    plots_dir: Path,
    experiment_data: dict[str, Any],
) -> list[str]:
    """Generate all applicable plots from experiment results. Returns list of saved paths."""
    saved = []

    if "recall_by_distance" in experiment_data:
        path = plots_dir / "recall_by_distance.png"
        plot_recall_by_distance(experiment_data["recall_by_distance"]["per_bin"], path)
        saved.append(str(path))

    if "lm_comparison" in experiment_data:
        for metric in ("loss", "perplexity"):
            if metric in experiment_data["lm_comparison"]:
                path = plots_dir / f"lm_{metric}_comparison.png"
                plot_comparison_bar(
                    experiment_data["lm_comparison"][metric],
                    path,
                    title=f"LM {metric.replace('_', ' ').title()} Comparison",
                    ylabel=metric,
                )
                saved.append(str(path))

    if "scale_curves" in experiment_data:
        sc = experiment_data["scale_curves"]
        path = plots_dir / "scale_curves.png"
        plot_scale_curves(sc["seq_lengths"], sc["metrics"], path, ylabel=sc.get("ylabel", "Metric"))
        saved.append(str(path))

    if "metrics_history" in experiment_data:
        history = experiment_data["metrics_history"]
        tags = [
            "task/lm_loss",
            "task/routing_loss",
            "task/lm_vs_random",
            "transformer/lm_loss",
            "transformer/total_loss",
            "router/infonce_loss",
            "train/loss",
            "eval/recall@32",
            "eval/loss",
        ]
        tags.extend(sorted(k for k in history if k.startswith("router/layer_")))
        seen = set()
        for tag in tags:
            if tag in history and tag not in seen:
                seen.add(tag)
                safe = tag.replace("/", "_")
                path = plots_dir / f"{safe}_curve.png"
                plot_training_curve(history, tag, path)
                saved.append(str(path))

    return saved
