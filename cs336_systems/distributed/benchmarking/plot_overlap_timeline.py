"""
Clear schematic for NaiveDDP vs OverlapDDP.

The left panel shows why overlap helps: communication moves from a separate
post-backward phase into the backward window. The right panel anchors the
schematic with the measured end-to-end step time.

Output: reports/figures/overlap_timeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

COMPUTE = "#22a06b"
COMM = "#d94848"
OTHER = "#d7d7d2"
HIDDEN = "#f4b7b7"
SURFACE = "#fcfcfb"
INK = "#111111"
MUTED = "#6f6f6a"
GRID = "#e6e3de"

OUT_PATH = Path("reports/figures/overlap_timeline.png")

# Measured timings (seconds) from benchmark_overlap_ddp.py/report section 7.2.
N_BWD = 0.295
N_SYNC = 0.398
N_TOTAL = 1.122

O_BWD = 0.531
O_WAIT = 0.020
O_TOTAL = 0.977

# Schematic placement for async all_reduce. The exact trace contains many small
# kernels; a single band makes the overlap relationship legible.
O_COMM_START = 0.080
O_COMM_END = O_BWD + O_WAIT

BAR_H = 0.30
LANE_GAP = 0.32


def main() -> None:
    fig = plt.figure(figsize=(12.8, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.05], wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax_result = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax_result.set_facecolor(SURFACE)

    draw_overlap_panel(ax)
    draw_result_panel(ax_result)

    fig.suptitle(
        "OverlapDDP hides gradient communication inside backward",
        fontsize=17,
        fontweight="bold",
        color=INK,
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "xl model, 2 GPUs, batch size 4 - measured with the same benchmark settings",
        ha="center",
        fontsize=10.5,
        color=MUTED,
    )

    handles = [
        Patch(facecolor=COMPUTE, label="backward compute window"),
        Patch(facecolor=COMM, label="all_reduce communication"),
        Patch(facecolor=HIDDEN, label="communication hidden by compute"),
        Patch(facecolor=OTHER, label="other step work"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.035),
    )

    fig.subplots_adjust(left=0.12, right=0.97, top=0.835, bottom=0.205, wspace=0.27)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


def draw_overlap_panel(ax: plt.Axes) -> None:
    naive_compute_y = 3.20
    naive_comm_y = naive_compute_y - LANE_GAP
    overlap_compute_y = 1.30
    overlap_comm_y = overlap_compute_y - LANE_GAP

    # NaiveDDP: backward finishes before communication starts.
    add_bar(ax, 0, N_BWD, naive_compute_y, COMPUTE, "backward\n295 ms")
    add_bar(ax, N_BWD, N_SYNC, naive_comm_y, COMM, "gradient sync\n398 ms")
    ax.plot(
        [N_BWD, N_BWD],
        [naive_comm_y - BAR_H / 2 - 0.05, naive_compute_y + BAR_H / 2 + 0.05],
        color=MUTED,
        linewidth=1.0,
        linestyle=":",
    )
    ax.text(
        N_BWD + 0.012,
        naive_compute_y + 0.30,
        "comm starts only\nafter backward",
        fontsize=9.5,
        color=MUTED,
        va="bottom",
    )

    # OverlapDDP: communication is launched by grad hooks while backward runs.
    add_bar(ax, 0, O_BWD, overlap_compute_y, COMPUTE, "backward window\n531 ms")
    add_bar(ax, O_COMM_START, O_COMM_END - O_COMM_START, overlap_comm_y, COMM, "")

    hidden_width = O_BWD - O_COMM_START
    hidden_rect = Rectangle(
        (O_COMM_START, overlap_comm_y - BAR_H / 2),
        hidden_width,
        BAR_H,
        facecolor=HIDDEN,
        edgecolor=COMM,
        hatch="////",
        linewidth=0.0,
        alpha=0.95,
        zorder=4,
    )
    ax.add_patch(hidden_rect)
    ax.text(
        O_COMM_START + hidden_width / 2,
        overlap_comm_y,
        "hidden by\nbackward",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    add_bar(ax, O_BWD, O_WAIT, overlap_comm_y, COMM, "", alpha=1.0)
    ax.annotate(
        "20 ms exposed wait",
        xy=(O_BWD + O_WAIT / 2, overlap_comm_y),
        xytext=(O_BWD + 0.030, overlap_comm_y - 0.36),
        arrowprops={"arrowstyle": "-", "color": COMM, "linewidth": 1.2},
        fontsize=9.5,
        color=COMM,
        fontweight="bold",
        ha="left",
    )

    saved = (N_BWD + N_SYNC) - (O_BWD + O_WAIT)
    ax.annotate(
        f"{saved * 1000:.0f} ms less exposed\nbackward+sync time",
        xy=(O_COMM_END, 0.26),
        xytext=(N_BWD + N_SYNC - saved / 2, 0.26),
        arrowprops={"arrowstyle": "<->", "color": COMM, "linewidth": 1.4},
        fontsize=10.5,
        fontweight="bold",
        color=COMM,
        ha="center",
    )

    ax.text(-0.035, naive_compute_y, "NaiveDDP", ha="right", va="center", fontsize=12, fontweight="bold", color=INK, clip_on=False)
    ax.text(-0.035, naive_comm_y, "serial", ha="right", va="center", fontsize=9.5, color=MUTED, clip_on=False)
    ax.text(-0.035, overlap_compute_y, "OverlapDDP", ha="right", va="center", fontsize=12, fontweight="bold", color=INK, clip_on=False)
    ax.text(-0.035, overlap_comm_y, "overlapped", ha="right", va="center", fontsize=9.5, color=MUTED, clip_on=False)

    ax.set_title("A. Where the communication goes", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=14)
    ax.set_xlabel("Time inside backward + gradient sync (seconds)", fontsize=10.5, color=INK)
    ax.set_xlim(0, 0.74)
    ax.set_ylim(-0.05, 3.85)
    ax.set_yticks([])
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    style_axes(ax)


def draw_result_panel(ax: plt.Axes) -> None:
    rows = [1.0, 0.0]
    labels = ["NaiveDDP", "OverlapDDP"]
    totals = [N_TOTAL, O_TOTAL]
    other = [N_TOTAL - N_BWD - N_SYNC, O_TOTAL - O_BWD - O_WAIT]
    exposed = [N_BWD + N_SYNC, O_BWD + O_WAIT]

    for y, label, total, other_time, exposed_time in zip(rows, labels, totals, other, exposed):
        add_bar(ax, 0, other_time, y, OTHER, f"other\n{other_time * 1000:.0f} ms", text_color=INK, fontsize=8.8)
        add_bar(ax, other_time, exposed_time, y, COMM if label == "NaiveDDP" else COMPUTE, "", alpha=0.90)
        ax.text(total + 0.025, y, f"{total:.3f} s", va="center", ha="left", fontsize=12, fontweight="bold", color=INK)

    saved = N_TOTAL - O_TOTAL
    ax.annotate(
        "",
        xy=(O_TOTAL, -0.30),
        xytext=(N_TOTAL, -0.30),
        arrowprops={"arrowstyle": "<->", "color": COMM, "linewidth": 1.5},
    )
    ax.text(
        (N_TOTAL + O_TOTAL) / 2,
        -0.47,
        f"saved {saved * 1000:.0f} ms\n{saved / N_TOTAL * 100:.0f}% faster",
        ha="center",
        va="top",
        fontsize=11,
        color=COMM,
        fontweight="bold",
    )

    ax.set_title("B. End-to-end step time", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=14)
    ax.set_xlim(0, 1.24)
    ax.set_ylim(-0.70, 1.45)
    ax.set_yticks(rows)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("Training step time (seconds)", fontsize=10.5, color=INK)
    ax.set_xticks([0, 0.4, 0.8, 1.2])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    style_axes(ax)


def add_bar(
    ax: plt.Axes,
    left: float,
    width: float,
    y: float,
    color: str,
    label: str,
    *,
    alpha: float = 0.95,
    text_color: str = "white",
    fontsize: float = 10.5,
) -> None:
    ax.barh(y, width, BAR_H, left=left, color=color, alpha=alpha, edgecolor="none", zorder=3)
    if label and width >= 0.06:
        ax.text(
            left + width / 2,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=text_color,
            zorder=5,
        )


def style_axes(ax: plt.Axes) -> None:
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


if __name__ == "__main__":
    main()
