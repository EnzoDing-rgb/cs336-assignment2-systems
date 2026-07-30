"""
Plot all_reduce latency from CSV → PNG.

Reads artifacts/all_reduce_single_node.csv, writes reports/figures/all_reduce_latency.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Palette (dataviz reference, light mode)
# ---------------------------------------------------------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
PRIMARY_INK = "#0b0b0b"

# Map world_size → style
STYLE = {
    2: {"color": BLUE, "marker": "o", "label": "N = 2"},
    4: {"color": ORANGE, "marker": "s", "label": "N = 4"},
}

CSV_PATH = Path("artifacts/all_reduce_single_node.csv")
OUT_PATH = Path("reports/figures/all_reduce_latency.png")


def load(csv_path: Path) -> dict[int, list[tuple[int, float]]]:
    """Return {world_size: [(size_mb, ms_mean), ...]} sorted by size_mb."""
    data: dict[int, list[tuple[int, float]]] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            ws = int(row["world_size"])
            mb = int(row["size_mb"])
            ms = float(row["latency_ms_mean"])
            data.setdefault(ws, []).append((mb, ms))
    for pts in data.values():
        pts.sort(key=lambda x: x[0])
    return data


def plot(data: dict[int, list[tuple[int, float]]]) -> plt.Figure:
    """Build the latency chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for ws, pts in sorted(data.items()):
        style = STYLE[ws]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            color=style["color"],
            marker=style["marker"],
            markersize=8,
            linewidth=2,
            markeredgewidth=0,
            label=style["label"],
            zorder=3,
        )

    # --- axes ---
    ax.set_xscale("log")
    ax.set_xlabel("data size per GPU (MB)", color=PRIMARY_INK, fontsize=11)
    ax.set_ylabel("latency (ms)", color=PRIMARY_INK, fontsize=11)
    ax.set_title(
        "NCCL all_reduce latency — single node, 4× RTX PRO 6000 Blackwell (PHB)",
        color=PRIMARY_INK, fontsize=13, pad=14,
    )

    # ticks
    ax.set_xticks([1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.tick_params(colors=MUTED, labelsize=10)

    # grid
    ax.grid(True, which="major", color=GRIDLINE, linewidth=0.5, linestyle="-")
    ax.grid(True, which="minor", color=GRIDLINE, linewidth=0.25, linestyle="-")

    # spines
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDLINE)
        spine.set_linewidth(0.5)

    # legend — always present for ≥ 2 series
    legend = ax.legend(
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRIDLINE,
        fontsize=10,
        loc="upper left",
    )
    for text in legend.get_texts():
        text.set_color(PRIMARY_INK)

    fig.tight_layout(pad=1.5)
    return fig


def main() -> None:
    data = load(CSV_PATH)
    fig = plot(data)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
