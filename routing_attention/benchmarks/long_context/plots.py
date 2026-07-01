"""Plotting utilities for long-context retrieval benchmark."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from routing_attention.benchmarks.long_context.evaluation import EvalSummary


def _cell_accuracy(
    summary: EvalSummary,
    task_type: str,
    context_lengths: list[int],
    needle_depths: list[float],
) -> np.ndarray:
    grid = np.full((len(needle_depths), len(context_lengths)), np.nan, dtype=np.float64)
    for i, depth in enumerate(needle_depths):
        for j, length in enumerate(context_lengths):
            key = f"{task_type}|T={length}|d={depth}"
            if key in summary.by_cell:
                grid[i, j] = summary.by_cell[key]
            else:
                vals = [
                    r.correct
                    for r in summary.records
                    if r.task_type == task_type
                    and r.context_length == length
                    and abs(r.needle_depth - depth) < 1e-6
                ]
                if vals:
                    grid[i, j] = sum(vals) / len(vals)
    return grid


def plot_accuracy_heatmap(
    summary: EvalSummary,
    task_type: str,
    context_lengths: list[int],
    needle_depths: list[float],
    output_path: Path,
    title: str | None = None,
) -> None:
    grid = _cell_accuracy(summary, task_type, context_lengths, needle_depths)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        grid,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        xticklabels=[str(t) for t in context_lengths],
        yticklabels=[f"{d:.0%}" for d in needle_depths],
        ax=ax,
    )
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Needle Depth")
    ax.set_title(title or f"Exact Match — {task_type}")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_all_task_heatmaps(
    summary: EvalSummary,
    context_lengths: list[int],
    needle_depths: list[float],
    output_dir: Path,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    task_types = sorted({r.task_type for r in summary.records})
    for task in task_types:
        path = output_dir / f"heatmap_{task}.png"
        plot_accuracy_heatmap(summary, task, context_lengths, needle_depths, path)
        saved.append(str(path))
    return saved


def plot_overall_by_length(
    summary: EvalSummary,
    output_path: Path,
) -> None:
    lengths = sorted(summary.by_context_length.keys())
    values = [summary.by_context_length[l] for l in lengths]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lengths, values, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Context Length")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Accuracy vs Context Length")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_overall_by_depth(
    summary: EvalSummary,
    output_path: Path,
) -> None:
    depths = sorted(summary.by_needle_depth.keys(), key=float)
    values = [summary.by_needle_depth[d] for d in depths]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"{float(d):.0%}" for d in depths], values, color="steelblue", alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Needle Depth")
    ax.set_ylabel("Exact Match Accuracy")
    ax.set_title("Accuracy vs Needle Depth")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_all_benchmark_plots(
    summary: EvalSummary,
    context_lengths: list[int],
    needle_depths: list[float],
    output_dir: Path,
) -> list[str]:
    output_dir = Path(output_dir)
    saved = plot_all_task_heatmaps(summary, context_lengths, needle_depths, output_dir)
    by_len = output_dir / "accuracy_by_context_length.png"
    plot_overall_by_length(summary, by_len)
    saved.append(str(by_len))
    by_depth = output_dir / "accuracy_by_needle_depth.png"
    plot_overall_by_depth(summary, by_depth)
    saved.append(str(by_depth))
    return saved
