"""
Overlap timeline: clean Gantt + kernel-density comparison.

Upper panel — schematic Gantt from benchmark timings (NaiveDDP vs OverlapDDP).
Lower panel — kernel activity density from nsys SQLite (histogram, 5ms bins).

Output: reports/figures/overlap_timeline.png
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BLUE    = "#2a78d6"
ORANGE  = "#eb6834"
GREEN   = "#1baf7a"
RED     = "#e34948"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
MUTED    = "#898781"
INK      = "#0b0b0b"

COMPUTE_COLOR = GREEN
COMM_COLOR    = RED

NAIVE_SQLITE   = Path("artifacts/nsys_naive.sqlite")
OVERLAP_SQLITE = Path("artifacts/nsys_overlap.sqlite")
OUT_PATH       = Path("reports/figures/overlap_timeline.png")

# Benchmark timing numbers (seconds) — from benchmark_overlap_ddp.py
NAIVE_BWD  = 0.295
NAIVE_SYNC = 0.398
OVERLAP_BWD  = 0.531
OVERLAP_SYNC = 0.020

BIN_MS = 5  # 5 ms bins for kernel density


# ---------------------------------------------------------------------------
# Kernel density from nsys SQLite
# ---------------------------------------------------------------------------
def kernel_density(sqlite_path: Path, t0_ns: int, t1_ns: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers_s, compute_density, comm_density) for one step."""
    db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    rows = db.execute("""
        SELECT start, end, demangledName
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE demangledName IS NOT NULL
          AND start >= ? AND end <= ?
        ORDER BY start
    """, (t0_ns, t1_ns)).fetchall()
    db.close()

    duration_s = (t1_ns - t0_ns) / 1e9
    n_bins = max(1, int(duration_s / (BIN_MS / 1000)))
    compute_hist = np.zeros(n_bins)
    comm_hist    = np.zeros(n_bins)

    for start_ns, end_ns, name in rows:
        t0 = (start_ns - t0_ns) / 1e9
        t1 = (end_ns   - t0_ns) / 1e9
        bin_start = int(t0 / (BIN_MS / 1000))
        bin_end   = int(t1 / (BIN_MS / 1000))
        for b in range(max(0, bin_start), min(n_bins, bin_end + 1)):
            if "nccl" in str(name).lower():
                comm_hist[b] += 1
            else:
                compute_hist[b] += 1

    centers = (np.arange(n_bins) + 0.5) * (BIN_MS / 1000)
    return centers, compute_hist, comm_hist


def _find_step_window(sqlite_path: Path) -> tuple[int, int]:
    """Find time window around the second measurement step (after warmup).

    Returns (start_ns, end_ns) for one representative step.
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
        return 0, 1

    all_start  = min(r[0] for r in rows)
    all_end    = max(r[1] for r in rows)
    total_s    = (all_end - all_start) / 1e9

    # Take the middle 1.5 seconds (one step ≈ 1.1s for Naive, 1.0s for Overlap)
    mid = (all_start + all_end) // 2
    half_window = int(0.9e9)  # ±0.9s
    return mid - half_window, mid + half_window


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------
def plot_gantt(ax, y, label, bwd_s, sync_s, comm_start_s, comm_end_s):
    """Draw a single Gantt row: compute bar + communication bar."""
    bar_h = 0.42

    # Compute (backward)
    ax.barh(y, bwd_s, bar_h, left=0, color=COMPUTE_COLOR, alpha=0.85,
            zorder=3, label="Compute (backward)" if y == 1 else "")
    # Communication
    ax.barh(y, comm_end_s - comm_start_s, bar_h, left=comm_start_s,
            color=COMM_COLOR, alpha=0.85, zorder=4,
            label="Communication (all_reduce)" if y == 1 else "")
    # Gradient sync tail (only if comm extends past backward)
    tail = comm_end_s - bwd_s
    if tail > 0.005:
        ax.barh(y, tail, bar_h, left=bwd_s, color=COMM_COLOR, alpha=0.35,
                zorder=4, hatch="///")

    # Annotations
    ax.text(bwd_s / 2, y, f"backward\n{bwd_s:.3f}s", ha="center", va="center",
            fontsize=10, color="white", fontweight="bold")
    comm_mid = (comm_start_s + comm_end_s) / 2
    comm_w = comm_end_s - comm_start_s
    if comm_w > 0.06:
        ax.text(comm_mid, y, f"comm\n{comm_w:.3f}s", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")
    # Total label
    total = bwd_s + max(0, tail) + sync_s  # rough total
    ax.text(bwd_s + max(0, tail) + 0.015, y, f"{label}", va="center",
            fontsize=11, fontweight="bold", color=INK)


def main() -> None:
    # ── Figure layout ──
    fig = plt.figure(figsize=(13, 7.5))
    fig.patch.set_facecolor(SURFACE)

    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.3], hspace=0.35)

    ax_gantt  = fig.add_subplot(gs[0])
    ax_naive  = fig.add_subplot(gs[1])
    ax_overlap = fig.add_subplot(gs[2])

    # ────────────────────────────────────────────────────────
    # Row 0: Gantt chart from benchmark numbers
    # ────────────────────────────────────────────────────────
    ax_gantt.set_facecolor(SURFACE)
    ax_gantt.set_title("Schematic (benchmark timings)", color=INK,
                       fontsize=12, fontweight="bold", loc="left")

    # NaiveDDP
    plot_gantt(ax_gantt, 1, "NaiveDDP", NAIVE_BWD, 0,
               NAIVE_BWD, NAIVE_BWD + NAIVE_SYNC)
    # OverlapDDP — comm starts after ~20% of backward, ends ~20ms after backward
    comm_start_o = OVERLAP_BWD * 0.15
    comm_end_o   = OVERLAP_BWD + OVERLAP_SYNC
    plot_gantt(ax_gantt, 0, "OverlapDDP", OVERLAP_BWD, OVERLAP_SYNC,
               comm_start_o, comm_end_o)

    ax_gantt.set_ylim(-0.7, 1.7)
    ax_gantt.set_yticks([0, 1])
    ax_gantt.set_yticklabels([])
    ax_gantt.set_xlim(0, NAIVE_BWD + NAIVE_SYNC + 0.12)
    ax_gantt.set_xlabel("Time (seconds)", color=INK, fontsize=9)
    ax_gantt.tick_params(colors=MUTED, labelsize=9)
    for spine in ax_gantt.spines.values():
        spine.set_edgecolor(GRIDLINE); spine.set_linewidth(0.5)

    # Labels on y-axis
    ax_gantt.text(-0.03, 1, "NaiveDDP", va="center", ha="right",
                  fontsize=11, fontweight="bold", color=INK, transform=ax_gantt.get_yaxis_transform())
    ax_gantt.text(-0.03, 0, "OverlapDDP", va="center", ha="right",
                  fontsize=11, fontweight="bold", color=INK, transform=ax_gantt.get_yaxis_transform())

    # Legend
    from matplotlib.patches import Patch
    ax_gantt.legend(
        handles=[
            Patch(facecolor=COMPUTE_COLOR, alpha=0.85, label="Compute (backward)"),
            Patch(facecolor=COMM_COLOR, alpha=0.85, label="Communication (all_reduce)"),
        ],
        fontsize=9, framealpha=0.9, edgecolor=GRIDLINE,
        loc="lower right", ncol=2,
    )

    # ────────────────────────────────────────────────────────
    # Row 1-2: Kernel activity density from nsys
    # ────────────────────────────────────────────────────────
    for ax, sqlite_path, label in [
        (ax_naive,   NAIVE_SQLITE,   "NaiveDDP — Nsight kernel activity density"),
        (ax_overlap, OVERLAP_SQLITE, "OverlapDDP — Nsight kernel activity density"),
    ]:
        ax.set_facecolor(SURFACE)
        t0_ns, t1_ns = _find_step_window(sqlite_path)
        centers, comp, comm = kernel_density(sqlite_path, t0_ns, t1_ns)

        if len(centers) > 0:
            # Shade compute area
            ax.fill_between(centers, 0, comp, color=COMPUTE_COLOR, alpha=0.35,
                            linewidth=0, label="Compute kernels")
            # Line for comm (on top, more visible)
            ax.fill_between(centers, 0, comm, color=COMM_COLOR, alpha=0.55,
                            linewidth=0, label="NCCL kernels")
            # Smooth comm line
            if np.max(comm) > 0:
                ax.plot(centers, comm, color=COMM_COLOR, linewidth=1.2, alpha=0.9)

        ax.set_title(label, color=INK, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("kernels / bin", color=INK, fontsize=8)
        ax.set_xlabel("Time relative to step start (s)", color=INK, fontsize=9)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRIDLINE); spine.set_linewidth(0.5)
        ax.legend(fontsize=8, framealpha=0.85, edgecolor=GRIDLINE,
                  loc="upper right")

    fig.suptitle(
        "Backward–Communication Overlap: NaiveDDP vs OverlapDDP\n"
        "xl model, 2× RTX PRO 6000 Blackwell, Nsight Systems trace",
        color=INK, fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout(pad=3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, edgecolor="none", bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
