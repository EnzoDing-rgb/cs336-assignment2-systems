"""
Timeline comparison: NaiveDDP (serial) vs OverlapDDP (overlapped).

Reads nsys SQLite exports, draws two time-aligned Gantt rows:
  - Compute kernels (green)
  - Communication/NCCL kernels (red)

Output: reports/figures/overlap_timeline.png
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BLUE   = "#2a78d6"
ORANGE = "#eb6834"
GREEN  = "#1baf7a"
RED    = "#e34948"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED    = "#898781"
INK      = "#0b0b0b"

# Colours for compute vs communication
COMPUTE_COLOR = GREEN
COMM_COLOR    = RED

NAIVE_SQLITE   = Path("artifacts/nsys_naive.sqlite")
OVERLAP_SQLITE = Path("artifacts/nsys_overlap.sqlite")
OUT_PATH       = Path("reports/figures/overlap_timeline.png")

STEP_DURATION_S = 2.0  # how many seconds of trace to show (around 1 step)


def _reldiff(a: int, b: int) -> float:
    return abs(a - b) / max(abs(a), abs(b))


def load_kernels(sqlite_path: Path) -> list[tuple[float, float, str]]:
    """Return [(start_s, end_s, kind)] where kind is 'compute' or 'comm'.

    Times are relative to the first kernel in the trace (seconds).
    """
    db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    rows = db.execute("""
        SELECT start, end, demangledName
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE demangledName IS NOT NULL
        ORDER BY start
    """).fetchall()
    db.close()

    if not rows:
        return []

    t0 = min(r[0] for r in rows)
    kernels = []
    for start_ns, end_ns, name in rows:
        t_start = (start_ns - t0) / 1e9
        t_end   = (end_ns   - t0) / 1e9
        kind = "comm" if ("nccl" in str(name).lower()) else "compute"
        kernels.append((t_start, t_end, kind))
    return kernels


def plot_timeline(ax: plt.Axes, kernels: list[tuple[float, float, str]],
                  title: str, t_max: float):
    """Draw a Gantt-like timeline on ax."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold")

    # Draw each kernel as a thin horizontal span
    for t_start, t_end, kind in kernels:
        if t_start > t_max:
            break
        if t_end - t_start < 1e-6:  # skip < 1μs kernels (too thin)
            continue
        color = COMM_COLOR if kind == "comm" else COMPUTE_COLOR
        ax.axvspan(t_start, min(t_end, t_max), alpha=0.55, color=color,
                   linewidth=0, zorder=2 if kind == "comm" else 1)

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("Time (seconds)", color=INK, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDLINE)
        spine.set_linewidth(0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=COMPUTE_COLOR, alpha=0.7, label="Compute (matmul, elemwise, …)"),
        Patch(facecolor=COMM_COLOR, alpha=0.7, label="Communication (NCCL all_reduce)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.85,
              edgecolor=GRIDLINE, loc="upper right")


def main() -> None:
    naive_kerns  = load_kernels(NAIVE_SQLITE)
    overlap_kerns = load_kernels(OVERLAP_SQLITE)

    if not naive_kerns or not overlap_kerns:
        print("[plot] ERROR: missing nsys data — run nsys profile first")
        return

    # Find a time window containing one training step.
    # We take the second half of the trace (after warmup).
    t_max_naive  = naive_kerns[-1][1]
    t_max_overlap = overlap_kerns[-1][1]
    t_mid_naive  = t_max_naive * 0.45
    t_mid_overlap = t_max_overlap * 0.45

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5.5))
    fig.patch.set_facecolor(SURFACE)

    plot_timeline(ax1, naive_kerns, "NaiveDDP — backward then all_reduce (serial)",
                  t_max_naive)
    plot_timeline(ax2, overlap_kerns, "OverlapDDP — all_reduce during backward (overlapped)",
                  t_max_overlap)

    # Add alignment markers
    for ax in (ax1, ax2):
        # Mark warmup ↔ measured boundary (approximate)
        pass

    fig.suptitle(
        "Nsight Timeline: NaiveDDP vs OverlapDDP — 2× RTX PRO 6000 Blackwell, xl model",
        color=INK, fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=2)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
