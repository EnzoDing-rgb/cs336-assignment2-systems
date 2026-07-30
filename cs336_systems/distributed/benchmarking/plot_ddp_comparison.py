"""
Plot NaiveDDP vs FlattenDDP comparison — grouped bar chart.

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
RED = "#e34948"
PURPLE = "#4a3aa7"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
PRIMARY_INK = "#0b0b0b"

CSV_PATH = Path("artifacts/ddp_comparison.csv")
OUT_PATH = Path("reports/figures/ddp_gradient_sync_comparison.png")


def load_means() -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {seg: [] for seg in ["forward", "backward", "gradient_sync", "optimizer", "total"]}
    )
    with CSV_PATH.open() as f:
        for row in csv.DictReader(f):
            if int(row["batch_size"]) != 4:
                continue
            mode = row["mode"]
            groups[mode]["forward"].append(float(row["forward_s"]))
            groups[mode]["backward"].append(float(row["backward_s"]))
            groups[mode]["gradient_sync"].append(float(row["gradient_sync_s"]))
            groups[mode]["optimizer"].append(float(row["optimizer_s"]))
            groups[mode]["total"].append(float(row["total_s"]))
    return {
        mode: {seg: float(np.mean(vals)) for seg, vals in d.items()}
        for mode, d in groups.items()
    }


def plot(means: dict[str, dict[str, float]]) -> plt.Figure:
    fig, (ax_main, ax_total) = plt.subplots(1, 2, figsize=(12, 5.2),
                                              width_ratios=[3, 1])
    fig.patch.set_facecolor(SURFACE)

    # --- Grouped bars ---
    categories = [
        ("gradient_sync", "Gradient\nsync", RED),
        ("forward",        "Forward",       BLUE),
        ("backward",       "Backward",      ORANGE),
        ("optimizer",      "Optimizer",     PURPLE),
    ]

    x = np.arange(len(categories))
    width = 0.32

    naive_vals  = [means["naive"][key]  for key, _, _ in categories]
    flatten_vals = [means["flatten"][key] for key, _, _ in categories]

    for ax in (ax_main, ax_total):
        ax.set_facecolor(SURFACE)

    bars1 = ax_main.bar(x - width/2, naive_vals, width,
                        color=BLUE, alpha=0.88, zorder=3, label="NaiveDDP (291× all_reduce)")
    bars2 = ax_main.bar(x + width/2, flatten_vals, width,
                        color=ORANGE, alpha=0.88, zorder=3, label="FlattenDDP (1× all_reduce)")

    # Value labels on bars (NaiveDDP above, FlattenDDP below when taller)
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        v1, v2 = naive_vals[i], flatten_vals[i]
        diff = v2 - v1
        # NaiveDDP label
        ax_main.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.008,
                     f"{v1:.3f}", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=BLUE)
        # FlattenDDP label
        ax_main.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.008,
                     f"{v2:.3f}", ha="center", va="bottom", fontsize=9,
                     fontweight="bold", color=ORANGE)
        # Delta annotation on gradient_sync
        if categories[i][0] == "gradient_sync":
            y_mid = max(v1, v2) + 0.035
            ax_main.annotate(
                f"+{diff*1000:.0f} ms\n(+{diff/v1*100:.0f}%)",
                xy=(i, y_mid - 0.02), fontsize=10, fontweight="bold",
                color=RED, ha="center",
            )

    ax_main.set_xticks(x)
    ax_main.set_xticklabels([label for _, label, _ in categories], fontsize=10, color=PRIMARY_INK)
    ax_main.set_ylabel("Time (seconds)", color=PRIMARY_INK, fontsize=10)
    ax_main.set_title("Segment Breakdown", color=PRIMARY_INK, fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=9, framealpha=0.9, edgecolor=GRIDLINE)
    ax_main.grid(axis="y", color=GRIDLINE, linewidth=0.5, alpha=0.7)
    ax_main.tick_params(colors=MUTED, labelsize=9)
    for spine in ax_main.spines.values():
        spine.set_edgecolor(GRIDLINE)
        spine.set_linewidth(0.5)

    # --- Total bar ---
    total_naive = means["naive"]["total"]
    total_flatten = means["flatten"]["total"]
    total_diff = total_flatten - total_naive

    bars_t = ax_total.bar([0], [total_naive], width * 1.5,
                          color=BLUE, alpha=0.88, zorder=3)
    bars_t2 = ax_total.bar([1], [total_flatten], width * 1.5,
                           color=ORANGE, alpha=0.88, zorder=3)
    ax_total.bar_label(bars_t, [f"{total_naive:.3f}s"], fontsize=10,
                       fontweight="bold", color=BLUE, padding=5)
    ax_total.bar_label(bars_t2, [f"{total_flatten:.3f}s"], fontsize=10,
                       fontweight="bold", color=ORANGE, padding=5)
    ax_total.annotate(
        f"+{total_diff*1000:.0f} ms\n(+{total_diff/total_naive*100:.0f}%)",
        xy=(1, total_flatten + 0.04), fontsize=12, fontweight="bold",
        color=RED, ha="center",
    )

    ax_total.set_xticks([0, 1])
    ax_total.set_xticklabels(["NaiveDDP", "FlattenDDP"], fontsize=9, color=PRIMARY_INK)
    ax_total.set_title("Total Step", color=PRIMARY_INK, fontsize=12, fontweight="bold")
    ax_total.set_ylim(0, max(total_naive, total_flatten) * 1.18)
    ax_total.grid(axis="y", color=GRIDLINE, linewidth=0.5, alpha=0.7)
    ax_total.tick_params(colors=MUTED, labelsize=9)
    for spine in ax_total.spines.values():
        spine.set_edgecolor(GRIDLINE)
        spine.set_linewidth(0.5)

    fig.suptitle(
        "DDP Gradient Sync: NaiveDDP vs FlattenDDP — xl model, 2× RTX PRO 6000 Blackwell, batch=4",
        color=PRIMARY_INK, fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout(pad=2)
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
