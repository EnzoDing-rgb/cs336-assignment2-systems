"""
Optimizer State Sharding: memory savings vs speed tradeoff.

Two-panel figure: memory peak (left) + optimizer step time (right).
Output: reports/figures/sharded_optimizer_comparison.png
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Palette
BLUE    = "#2a78d6"
GREEN   = "#1baf7a"
RED     = "#e34948"
SURFACE = "#fcfcfb"
GRID    = "#e1e0d9"
MUTED   = "#898781"
INK     = "#0b0b0b"

CSV_PATH = Path("artifacts/sharded_optimizer_benchmark.csv")
OUT_PATH = Path("reports/figures/sharded_optimizer_comparison.png")


def load() -> dict[str, dict[str, float]]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            m = r["mode"]
            groups[m]["peak_before_opt"].append(float(r["peak_before_opt_mib"]))
            groups[m]["mem_before_opt"].append(float(r["mem_before_opt_mib"]))
            groups[m]["optimizer_s"].append(float(r["optimizer_s"]))
            groups[m]["total_s"].append(float(r["total_s"]))
            groups[m]["gradient_sync_s"].append(float(r["gradient_sync_s"]))
            groups[m]["backward_s"].append(float(r["backward_s"]))
            groups[m]["forward_s"].append(float(r["forward_s"]))
    return {
        m: {k: statistics.mean(v) for k, v in d.items()}
        for m, d in groups.items()
    }


def main() -> None:
    d = load()
    b, s = d["baseline"], d["sharded"]

    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    fig.patch.set_facecolor(SURFACE)

    # ── Left: Peak memory ──
    bar_w = 0.55
    x = [0, 1]
    mem_vals = [b["peak_before_opt"] / 1024, s["peak_before_opt"] / 1024]  # GiB
    colors = [BLUE, GREEN]

    bars = ax_mem.bar(x, mem_vals, bar_w, color=colors, alpha=0.88, zorder=3,
                      edgecolor="none")
    for xi, val, color in zip(x, mem_vals, colors):
        ax_mem.text(xi, val + 1.2, f"{val:.1f} GiB", ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=color)
    # Savings annotation
    saved = mem_vals[0] - mem_vals[1]
    ax_mem.annotate(
        f"−{saved:.1f} GiB\n(−{saved/mem_vals[0]*100:.0f}%)",
        xy=(0.5, min(mem_vals) + saved * 0.4),
        fontsize=13, fontweight="bold", color=RED, ha="center",
    )

    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels(["AdamW\n(baseline)", "ShardedOptimizer\n(2-way shard)"],
                           fontsize=11, color=INK)
    ax_mem.set_ylabel("Peak GPU memory (GiB)", fontsize=11, color=INK)
    ax_mem.set_title("Peak Memory  →  optimizer step", fontsize=12,
                     fontweight="bold", color=INK, loc="left")
    ax_mem.set_ylim(0, max(mem_vals) * 1.15)
    ax_mem.set_facecolor(SURFACE)
    ax_mem.tick_params(colors=MUTED, labelsize=10)
    ax_mem.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    for spine in ax_mem.spines.values():
        spine.set_edgecolor(GRID); spine.set_linewidth(0.5)

    # ── Right: Optimizer step time ──
    opt_times = [b["optimizer_s"] * 1000, s["optimizer_s"] * 1000]
    bars2 = ax_time.bar(x, opt_times, bar_w, color=colors, alpha=0.88, zorder=3,
                        edgecolor="none")
    for xi, val, color in zip(x, opt_times, colors):
        ax_time.text(xi, val + 5, f"{val:.0f} ms", ha="center", va="bottom",
                     fontsize=13, fontweight="bold", color=color)
    # Slowdown annotation
    slowdown = opt_times[1] - opt_times[0]
    pct = slowdown / opt_times[0] * 100
    ax_time.annotate(
        f"+{slowdown:.0f} ms\n(+{pct:.0f}%)",
        xy=(0.5, max(opt_times) - slowdown * 0.35),
        fontsize=13, fontweight="bold", color=RED, ha="center",
    )
    ax_time.text(0.5, max(opt_times) * 0.15,
                 "broadcast\n291 parameters",
                 ha="center", fontsize=10, color=RED, fontweight="bold")

    ax_time.set_xticks(x)
    ax_time.set_xticklabels(["AdamW\n(baseline)", "ShardedOptimizer\n(2-way shard)"],
                            fontsize=11, color=INK)
    ax_time.set_ylabel("Optimizer step time (ms)", fontsize=11, color=INK)
    ax_time.set_title("Optimizer Step  →  speed cost", fontsize=12,
                      fontweight="bold", color=INK, loc="left")
    ax_time.set_ylim(0, max(opt_times) * 1.25)
    ax_time.set_facecolor(SURFACE)
    ax_time.tick_params(colors=MUTED, labelsize=10)
    ax_time.grid(axis="y", color=GRID, linewidth=0.5, alpha=0.7)
    for spine in ax_time.spines.values():
        spine.set_edgecolor(GRID); spine.set_linewidth(0.5)

    # ── Title ──
    fig.suptitle(
        "Optimizer State Sharding — memory vs speed tradeoff\n"
        "xl model, 2× RTX PRO 6000 Blackwell, batch=4",
        fontsize=13, fontweight="bold", color=INK, y=1.03,
    )
    fig.tight_layout(pad=2.5, rect=(0, 0, 1, 0.94))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, facecolor=SURFACE, edgecolor="none",
                bbox_inches="tight")
    plt.close(fig)
    print(f"→ {OUT_PATH}")


if __name__ == "__main__":
    main()
