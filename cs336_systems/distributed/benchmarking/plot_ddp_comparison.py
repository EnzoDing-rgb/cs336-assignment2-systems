"""
Plot NaiveDDP vs FlattenDDP comparison chart.

Reads artifacts/naive_ddp_benchmark.csv, writes reports/figures/ddp_gradient_sync_comparison.png
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette (dataviz reference, light mode)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
RED = "#e34948"
PURPLE = "#4a3aa7"
MUTED = "#898781"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
PRIMARY_INK = "#0b0b0b"

SEGMENTS = ["forward", "loss", "backward", "gradient_sync", "optimizer"]
SEGMENT_COLORS = {
    "forward": BLUE,
    "loss": MUTED,
    "backward": ORANGE,
    "gradient_sync": RED,
    "optimizer": PURPLE,
}
SEGMENT_LABELS = {
    "forward": "Forward",
    "loss": "Loss",
    "backward": "Backward",
    "gradient_sync": "Gradient sync",
    "optimizer": "Optimizer",
}

CSV_PATH = Path("artifacts/naive_ddp_benchmark.csv")
OUT_PATH = Path("reports/figures/ddp_gradient_sync_comparison.png")


def load_means() -> dict[str, dict[str, float]]:
    """Return {mode: {segment: mean_seconds}} for batch=4."""
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {seg: [] for seg in SEGMENTS}
    )
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if int(row["batch_size"]) != 4:
                continue
            mode = row["mode"]
            groups[mode]["forward"].append(float(row["forward_s"]))
            groups[mode]["loss"].append(float(row["loss_s"]))
            groups[mode]["backward"].append(float(row["backward_s"]))
            groups[mode]["gradient_sync"].append(float(row["gradient_sync_s"]))
            groups[mode]["optimizer"].append(float(row["optimizer_s"]))

    return {
        mode: {seg: float(np.mean(vals)) for seg, vals in d.items()}
        for mode, d in groups.items()
    }


def plot(means: dict[str, dict[str, float]]) -> plt.Figure:
    modes = ["naive_ddp", "flatten_ddp"]
    labels = ["NaiveDDP\n(291 all_reduce)", "FlattenDDP\n(1 all_reduce)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(SURFACE)

    bar_order = ["forward", "loss", "backward", "gradient_sync", "optimizer"]

    # --- Left: stacked horizontal bars, side-by-side ---
    for ax, mode, label in zip([ax1, ax2], modes, labels):
        ax.set_facecolor(SURFACE)
        left = 0.0
        y_pos = 0
        total = sum(means[mode][seg] for seg in bar_order)

        for seg in bar_order:
            val = means[mode][seg]
            pct = (val / total) * 100
            color = SEGMENT_COLORS[seg]
            bar = ax.barh(y_pos, val, left=left, color=color, height=0.55, alpha=0.92, zorder=3)

            if pct > 6:
                ax.text(
                    left + val / 2, y_pos,
                    f"{SEGMENT_LABELS[seg]}\n{val:.3f}s ({pct:.0f}%)",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold",
                )
            left += val

        # Total label
        ax.text(left + 0.02, y_pos, f"{total:.3f}s", va="center",
                fontsize=11, fontweight="bold", color=PRIMARY_INK)

        # Highlight gradient sync portion
        sync_val = means[mode]["gradient_sync"]
        sync_pct = sync_val / total * 100
        ax.text(left + 0.02, y_pos - 0.45,
                f"sync: {sync_val:.3f}s ({sync_pct:.0f}%)",
                va="center", fontsize=10, color=RED)

        ax.set_xlim(0, left * 1.18)
        ax.set_ylim(-0.6, 0.6)
        ax.set_yticks([])
        ax.set_xlabel("Time (seconds)", color=PRIMARY_INK, fontsize=10)
        ax.set_title(label, color=PRIMARY_INK, fontsize=12, fontweight="bold")
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRIDLINE)
            spine.set_linewidth(0.5)

    # Shared legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=SEGMENT_COLORS[seg], alpha=0.92, label=SEGMENT_LABELS[seg])
        for seg in bar_order
    ]
    fig.legend(
        handles=legend_patches, loc="lower center", ncol=5,
        fontsize=9, framealpha=0.9, edgecolor=GRIDLINE,
    )

    fig.suptitle(
        "DDP Gradient Synchronization: NaiveDDP vs FlattenDDP\n"
        "xl model, 2× RTX PRO 6000 Blackwell, batch=4, 10 steps",
        color=PRIMARY_INK, fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(pad=2.5)
    fig.subplots_adjust(bottom=0.18)
    return fig


def main() -> None:
    means = load_means()
    fig = plot(means)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
