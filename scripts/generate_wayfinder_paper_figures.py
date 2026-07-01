#!/usr/bin/env python3
"""Generate figures for wayfinder_attention_paper_v0_4.tex into paper_figures/."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
RESULTS = ROOT / "experiments" / "Experiment_7" / "learned_address_breakthrough" / "results_table_full.json"
SYSTEMS = ROOT / "experiments" / "Experiment_7" / "learned_address_proof_cell" / "proof_cell_full" / "systems_benchmark.json"

DEPTH_LABELS = ["10%", "25%", "50%", "75%", "90%"]
DEPTH_KEYS = [
    "0.10049019607843138",
    "0.25",
    "0.49950980392156863",
    "0.7490196078431373",
    "0.8985294117647059",
]

# Paper display names and colors
VARIANT_STYLE = {
    "dense_flash": ("Dense Flash", "#2563eb"),
    "learned_address_k32": ("Wayfinder", "#059669"),
    "key_vector_k32": ("Key-vector", "#d97706"),
    "local_window64": ("Local W=64", "#dc2626"),
    "linear": ("Linear", "#7c3aed"),
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _proof_cell_row(results: dict, variant: str) -> dict:
    for row in results["quality_rows"]:
        if row.get("experiment") == "proof_cell" and row.get("variant") == variant:
            return row
    raise KeyError(variant)


def plot_depth_bars(results: dict, out: Path) -> None:
    variants = ["dense_flash", "learned_address_k32", "key_vector_k32", "local_window64"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(DEPTH_LABELS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for off, var in zip(offsets, variants):
        row = _proof_cell_row(results, var)
        vals = [100.0 * row["by_needle_depth"][k] for k in DEPTH_KEYS]
        label, color = VARIANT_STYLE[var]
        ax.bar([i + off * width for i in x], vals, width=width, label=label, color=color, edgecolor="white", linewidth=0.6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(DEPTH_LABELS)
    ax.set_xlabel("Needle depth")
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Proof cell @ T=2048 (seed 45, 0 decoys)")
    ax.legend(loc="lower left", ncol=2, frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out)
    plt.close(fig)


def _systems_map(systems: dict) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for row in systems["rows"]:
        if row.get("error"):
            continue
        out[(row["variant"], int(row["context_length"]))] = float(row["latency_ms"])
    return out


def _linear_latency_from_results(results: dict) -> dict[int, float]:
    """Linear rows from separate benchmark session (paper Table 4)."""
    by_len: dict[int, float] = {}
    for row in results.get("systems_rows", []):
        if row.get("variant") != "linear" or row.get("error"):
            continue
        ctx = int(row["context_length"])
        # Prefer first recorded row per length (earlier canonical-style run in aggregate file)
        if ctx not in by_len:
            by_len[ctx] = float(row["latency_ms"])
    return by_len


def plot_latency(systems: dict, results: dict, out: Path) -> None:
    lengths = [2048, 4096, 8192, 16384]
    sys_map = _systems_map(systems)
    linear_map = _linear_latency_from_results(results)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for var, (label, color) in VARIANT_STYLE.items():
        ys = []
        for t in lengths:
            if var == "linear":
                ys.append(linear_map.get(t))
            else:
                ys.append(sys_map.get((var, t)))
        if any(y is None for y in ys):
            continue
        ax.plot(lengths, ys, marker="o", linewidth=2, markersize=6, label=label, color=color)

    ax.set_xscale("log", base=2)
    ax.set_xticks(lengths)
    ax.set_xticklabels([str(t) for t in lengths])
    ax.set_xlabel("Sequence length (tokens)")
    ax.set_ylabel("Forward latency (ms)")
    ax.set_title("Prototype forward latency (batch size 1, 4L d=256)")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(out)
    plt.close(fig)


def plot_bsweep(results: dict, out: Path) -> None:
    rows = [
        r for r in results["quality_rows"]
        if r.get("experiment") == "phase1_sweep" and r.get("variant") == "learned_address_k32"
    ]
    rows.sort(key=lambda r: int(r["b_steps"]))
    steps = [int(r["b_steps"]) for r in rows]
    recall = [100.0 * float(r["recall_at_k_after_b"]) for r in rows]
    acc = [100.0 * float(r["accuracy"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax2 = ax1.twinx()
    l1 = ax1.plot(steps, acc, color="#059669", marker="s", linewidth=2, label="Phase C task accuracy")
    l2 = ax2.plot(steps, recall, color="#2563eb", marker="o", linewidth=2, linestyle="--", label="Recall@32 after Phase B")
    ax1.set_xlabel("Phase B steps per layer")
    ax1.set_ylabel("Task accuracy (%)", color="#059669")
    ax2.set_ylabel("Recall@32 (%)", color="#2563eb")
    ax1.set_ylim(95, 101)
    ax2.set_ylim(55, 65)
    ax1.set_title("Phase B budget sweep @ T=2048")
    ax1.grid(axis="x", alpha=0.25)
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)
    fig.savefig(out)
    plt.close(fig)


def _box(ax, xy, wh, text, fc="#f8fafc", ec="#334155", fontsize=9):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)


def _arrow(ax, p0, p1):
    arr = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=1.2,
        color="#475569", transform=ax.transAxes, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(arr)


def plot_protocol(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.22, 0.82), (0.56, 0.11), "Phase A: Dense teacher\n20k steps, restore best checkpoint", fc="#dbeafe")
    _arrow(ax, (0.5, 0.82), (0.5, 0.76))
    _box(ax, (0.22, 0.63), (0.56, 0.11), "Cache hidden states + teacher attention\n64 train / 16 holdout cache batches", fc="#e2e8f0")
    _arrow(ax, (0.5, 0.63), (0.5, 0.57))
    _box(ax, (0.22, 0.44), (0.56, 0.11), "Phase B: Address index pretrain\nInfoNCE over teacher top-32, 10k steps/layer", fc="#fef3c7")
    _arrow(ax, (0.5, 0.44), (0.5, 0.38))
    _box(ax, (0.22, 0.25), (0.56, 0.11), "Phase C: Sparse fine-tune\nTop-K=128 Wayfinder, addresses trainable, 20k steps", fc="#d1fae5")
    _arrow(ax, (0.5, 0.25), (0.5, 0.19))
    _box(ax, (0.22, 0.06), (0.56, 0.11), "Evaluate on 300-example holdout\nPointer-unique NIAH, seed 45", fc="#f1f5f9")

    ax.set_title("Three-phase Wayfinder training protocol", pad=12)
    fig.savefig(out)
    plt.close(fig)


def plot_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, (0.38, 0.86), (0.24, 0.08), "Hidden states H", fc="#e2e8f0", fontsize=10)
    _arrow(ax, (0.5, 0.86), (0.28, 0.78))
    _arrow(ax, (0.5, 0.86), (0.72, 0.78))

    _box(ax, (0.08, 0.62), (0.36, 0.14), "Address path (search)\n$a^Q = W_A^Q h$, $a^K = W_A^K h$\nTop-K=128 neighbor retrieval", fc="#fef3c7")
    _box(ax, (0.56, 0.62), (0.36, 0.14), "QKV path (aggregation)\n$Q,K,V$ projections\nSoftmax over retrieved set", fc="#dbeafe")

    _arrow(ax, (0.26, 0.62), (0.26, 0.48))
    _arrow(ax, (0.74, 0.62), (0.74, 0.48))
    _box(ax, (0.18, 0.36), (0.16, 0.10), "Candidate\nset $N_K(i)$", fc="#fde68a", fontsize=8)
    _box(ax, (0.66, 0.36), (0.16, 0.10), "Sparse\nsoftmax", fc="#bfdbfe", fontsize=8)

    _arrow(ax, (0.34, 0.41), (0.50, 0.28))
    _arrow(ax, (0.74, 0.36), (0.50, 0.28))
    _box(ax, (0.30, 0.14), (0.40, 0.12), "Wayfinder attention output\nSearch $\\circ$ aggregation", fc="#d1fae5", fontsize=10)

    ax.text(0.02, 0.02, "Asymmetric address similarity for retrieval; standard Q/K/V for reading values.", fontsize=8, color="#475569")
    ax.set_title("Wayfinder Attention: separate search from aggregation", pad=10)
    fig.savefig(out)
    plt.close(fig)


def plot_retrieval_tiling(out: Path) -> None:
    """Schematic: naive T×T materialization vs fused causal tile streaming."""
    T = 10
    K = 3
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[0.10, 1.0, 0.14],
        width_ratios=[1.05, 1.0, 0.62],
        hspace=0.08,
        wspace=0.06,
        left=0.05,
        right=0.98,
        top=0.90,
        bottom=0.08,
    )

    def draw_causal_grid(ax, mode: str) -> None:
        x_hi = T + 1.35 if mode == "naive" else T - 0.1
        ax.set_xlim(-0.8, x_hi)
        ax.set_ylim(-0.8, T - 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        for i in range(T):
            for j in range(i + 1):
                x, y = j, T - 1 - i
                if mode == "naive":
                    color = "#fecaca" if (i + j) % 2 == 0 else "#fca5a5"
                    alpha = 0.85
                else:
                    i0 = 3
                    in_block = i0 <= i < i0 + 4 and j <= i
                    in_tile = in_block and j < 7
                    if in_tile:
                        color, alpha = "#fde68a", 0.95
                    elif in_block:
                        color, alpha = "#fef3c7", 0.7
                    else:
                        color, alpha = "#f1f5f9", 0.35
                ax.add_patch(
                    mpatches.FancyBboxPatch(
                        (x, y), 0.92, 0.92,
                        boxstyle="square,pad=0",
                        linewidth=0.4,
                        edgecolor="#94a3b8",
                        facecolor=color,
                        alpha=alpha,
                    )
                )
        ax.text(T / 2 - 0.5, -0.55, "key index $j$", ha="center", fontsize=9)
        ax.text(-0.55, T / 2 - 0.5, "query $i$", va="center", rotation=90, fontsize=9)
        if mode == "naive":
            ax.text(
                T - 0.55, T / 2 - 0.5,
                "Materialize full\n$S \\in \\mathbb{R}^{T \\times T}$\nin HBM",
                ha="left", va="center", fontsize=8.5, color="#b91c1c",
            )
        if mode == "fused":
            bx, by = 0, T - 1 - 6
            ax.add_patch(
                mpatches.Rectangle(
                    (bx - 0.08, by - 0.08), 7.16, 4.16,
                    linewidth=2.2, edgecolor="#d97706", facecolor="none",
                )
            )
            ax.text(3.4, by + 4.35, "active row block $B_m$", ha="center", fontsize=8, color="#b45309")

    def title_ax(col_slice):
        ax = fig.add_subplot(gs[0, col_slice])
        ax.axis("off")
        return ax

    def footer_ax(col_slice):
        ax = fig.add_subplot(gs[2, col_slice])
        ax.axis("off")
        return ax

    # --- Naive panel ---
    ax_naive = fig.add_subplot(gs[1, 0])
    draw_causal_grid(ax_naive, "naive")

    ax_naive_title = title_ax(slice(0, 1))
    ax_naive_title.text(0.5, 0.5, "Naive retrieval", ha="center", va="center", fontsize=11, fontweight="medium")

    ax_naive_footer = footer_ax(slice(0, 1))
    ax_naive_footer.text(
        0.5, 0.65,
        "Peak workspace: $\\Theta(BT^2)$\nOutput: $\\mathcal{O}(BTK)$ indices",
        ha="center", va="center", fontsize=8.5, color="#334155",
    )

    # --- Fused panel ---
    ax_fused = fig.add_subplot(gs[1, 1])
    draw_causal_grid(ax_fused, "fused")

    ax_fused_title = title_ax(slice(1, 3))
    ax_fused_title.text(0.5, 0.5, "Fused causal retrieval", ha="center", va="center", fontsize=11, fontweight="medium")

    ax_fused_footer = footer_ax(slice(1, 3))
    ax_fused_footer.text(
        0.5, 0.65,
        "Peak workspace: $\\mathcal{O}(B \\cdot B_m \\cdot T)$\nOutput: $\\mathcal{O}(BTK)$ indices",
        ha="center", va="center", fontsize=8.5, color="#334155",
    )

    side_gs = gs[1, 2].subgridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.05)
    ax_fused_stream = fig.add_subplot(side_gs[0, 0])
    ax_fused_stream.axis("off")
    ax_fused_stream.text(
        0.5, 0.5, "Stream tiles;\nmerge top-$K$;\ndiscard scores",
        ha="center", va="center", fontsize=8.5, color="#059669",
    )

    ax_fused_topk = fig.add_subplot(side_gs[0, 1])
    ax_fused_topk.set_xlim(-0.2, K + 0.2)
    ax_fused_topk.set_ylim(-0.8, T - 0.1)
    ax_fused_topk.set_aspect("equal")
    ax_fused_topk.axis("off")
    for i in range(T):
        y = T - 1 - i
        for k in range(K):
            ax_fused_topk.add_patch(
                mpatches.Rectangle(
                    (k + 0.12, y + 0.12), 0.76, 0.76,
                    facecolor="#bbf7d0", edgecolor="#059669", linewidth=0.6,
                )
            )
    ax_fused_topk.text(
        (K - 1) / 2 + 0.5, T + 0.15,
        "top-$K$ indices", ha="center", fontsize=8, color="#047857",
    )

    fig.suptitle(
        "Address search: same exact causal top-$K$, lower HBM footprint via tiled fusion",
        fontsize=11, fontweight="medium", y=0.98,
    )
    fig.savefig(out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = _load_json(RESULTS)
    systems = _load_json(SYSTEMS)

    plot_depth_bars(results, OUT / "fig_depth_bars.png")
    plot_latency(systems, results, OUT / "fig_latency.png")
    plot_bsweep(results, OUT / "fig_bsweep.png")
    plot_protocol(OUT / "fig_protocol.png")
    plot_architecture(OUT / "fig_architecture.png")
    plot_retrieval_tiling(OUT / "fig_retrieval_tiling.png")

    print(f"Wrote figures to {OUT}:")
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
