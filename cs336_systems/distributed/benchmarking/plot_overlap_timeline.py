"""
Single-focus Gantt chart: NaiveDDP (serial) vs OverlapDDP (overlapped).

Two rows, two colour blocks — green = backward compute, red = all_reduce.
One clear message: OverlapDDP hides communication inside backward.

Output: reports/figures/overlap_timeline.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
COMPUTE = "#1baf7a"   # green — backward computation
COMM   = "#e34948"    # red   — all_reduce communication
SURFACE = "#fcfcfb"
INK     = "#0b0b0b"
MUTED   = "#898781"

OUT_PATH = Path("reports/figures/overlap_timeline.png")

# Measured timings (seconds) — from benchmark_overlap_ddp.py
N_BWD  = 0.295   # NaiveDDP backward
N_SYNC = 0.398   # NaiveDDP gradient sync
O_BWD  = 0.531   # OverlapDDP backward (includes overlapped comm)
O_WAIT = 0.020   # OverlapDDP wait-for-handles tail

# OverlapDDP: comm starts roughly 15% into backward (after first layers finish),
# and the last handle finishes ~20ms after backward ends.
O_COMM_START = 0.08   # 80ms into backward, first hook fires
O_COMM_END   = O_BWD + O_WAIT


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_h = 0.55
    y_naive  = 1.0
    y_overlap = 0.0

    # ── NaiveDDP row ──
    # Compute
    ax.barh(y_naive, N_BWD, bar_h, left=0, color=COMPUTE, alpha=0.92, zorder=3)
    # Communication
    ax.barh(y_naive, N_SYNC, bar_h, left=N_BWD, color=COMM, alpha=0.92, zorder=3)

    # Labels on bars
    ax.text(N_BWD / 2, y_naive, f"backward\n{N_BWD:.3f}s",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(N_BWD + N_SYNC / 2, y_naive, f"gradient sync\n{N_SYNC:.3f}s",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    # Total
    n_total = N_BWD + N_SYNC
    ax.text(n_total + 0.015, y_naive + 0.2, f"1.122 s",
            fontsize=12, fontweight="bold", color=INK)

    # ── OverlapDDP row ──
    # Compute (stretched because communication runs inside it)
    ax.barh(y_overlap, O_BWD, bar_h, left=0, color=COMPUTE, alpha=0.92, zorder=3)
    # Communication — starts early, ends just after backward
    ax.barh(y_overlap, O_COMM_END - O_COMM_START, bar_h,
            left=O_COMM_START, color=COMM, alpha=0.92, zorder=4)

    ax.text(O_BWD / 2, y_overlap, f"backward\n{O_BWD:.3f}s",
            ha="center", va="center", fontsize=11, color="white", fontweight="bold")

    # Communication label — place inside if wide enough, else above
    comm_w = O_COMM_END - O_COMM_START
    comm_mid = O_COMM_START + comm_w / 2
    ax.text(comm_mid, y_overlap, f"async\nall_reduce\n{comm_w:.3f}s",
            ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    # Wait tail annotation
    ax.text(O_BWD + 0.01, y_overlap + 0.35, f"wait\n{O_WAIT:.3f}s",
            ha="left", va="bottom", fontsize=9, color=COMM, fontweight="bold")

    # Total
    o_total = O_BWD + O_WAIT
    ax.text(o_total + 0.015, y_overlap + 0.2, f"0.977 s",
            fontsize=12, fontweight="bold", color=INK)

    # ── "Saved" arrow between totals ──
    saved = n_total - o_total
    ax.annotate(
        f"−{saved*1000:.0f} ms\n(−{saved/n_total*100:.0f}%)",
        xy=(o_total, y_overlap - 0.4),
        fontsize=11, fontweight="bold", color=COMM,
        ha="center",
    )

    # ── Axes ──
    x_max = n_total * 1.12
    ax.set_xlim(0, x_max)
    ax.set_ylim(-0.75, 1.75)
    ax.set_yticks([y_overlap, y_naive])
    ax.set_yticklabels(["OverlapDDP", "NaiveDDP"], fontsize=12,
                       fontweight="bold", color=INK)
    ax.set_xlabel("Time (seconds)", fontsize=11, color=INK)
    ax.tick_params(colors=MUTED, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Title ──
    ax.set_title(
        "Backward–Communication Overlap: NaiveDDP vs OverlapDDP\n"
        "xl model, 2× RTX PRO 6000 Blackwell, batch=4",
        fontsize=13, fontweight="bold", color=INK, pad=16,
    )

    # ── Legend ──
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=COMPUTE, alpha=0.92, label="backward computation"),
            Patch(facecolor=COMM,   alpha=0.92, label="all_reduce communication"),
        ],
        loc="lower right", fontsize=10, framealpha=0.9,
        edgecolor="#e0e0e0", ncol=2,
    )

    fig.tight_layout(pad=1.5)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, edgecolor="none",
                bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
