"""
FSDP: peak memory savings + step time comparison.
Master's style — gridspec, add_bar, style_axes, left-aligned titles.

Three panels: Peak memory (left), Step time breakdown (center), Speedup summary (right).
Output: reports/figures/fsdp_comparison.png
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Palette & constants ──
BLUE   = "#2a78d6"
GREEN  = "#22a06b"
TEAL   = "#1baf7a"
RED    = "#d94848"
OTHER  = "#d7d7d2"
SURFACE = "#fcfcfb"
INK    = "#111111"
MUTED  = "#6f6f6a"
GRID   = "#e6e3de"

BAR_H   = 0.30
LANE_GAP = 0.20

FSDP_CSV = Path("artifacts/fsdp_benchmark.csv")
SHARDED_CSV = Path("artifacts/sharded_optimizer_benchmark.csv")
OUT_PATH = Path("reports/figures/fsdp_comparison.png")

# DDP baseline from Section 8 (hardcoded for clarity)
DDP_PEAK_GIB = 52.1
DDP_TOTAL_S  = 1.122
DDP_OPT_S    = 0.274


def load():
    """Return {(mode, world_size): {metric: mean}}}."""
    groups = defaultdict(lambda: defaultdict(list))
    with FSDP_CSV.open() as f:
        for r in csv.DictReader(f):
            key = (r["mode"], int(r["world_size"]))
            groups[key]["peak"].append(float(r["peak_before_opt_mib"]) / 1024)
            groups[key]["mem"].append(float(r["mem_before_opt_mib"]) / 1024)
            groups[key]["total"].append(float(r["total_s"]))
            groups[key]["opt"].append(float(r["optimizer_s"]))
            groups[key]["fwd"].append(float(r["forward_s"]))
            groups[key]["bwd"].append(float(r["backward_s"]))
            groups[key]["sync"].append(float(r["gradient_sync_s"]))
    return {k: {mk: statistics.mean(v) for mk, v in d.items()}
            for k, d in groups.items()}


def add_bar(ax, left, width, y, color, label, *, alpha=0.95,
            text_color="white", fontsize=10.5):
    ax.barh(y, width, BAR_H, left=left, color=color, alpha=alpha,
            edgecolor="none", zorder=3)
    if label and width >= 0.04:
        ax.text(left + width / 2, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=5)


def style_axes(ax):
    ax.tick_params(colors=MUTED, labelsize=9.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    d = load()
    configs = [
        ("DDP\nbaseline", None),
        ("FSDP 2-GPU\nFP32", ("fsdp_fp32", 2)),
        ("FSDP 2-GPU\nBF16", ("fsdp_bf16", 2)),
        ("FSDP 4-GPU\nFP32", ("fsdp_fp32", 4)),
        ("FSDP 4-GPU\nBF16", ("fsdp_bf16", 4)),
    ]

    fig = plt.figure(figsize=(13.5, 5.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.9], wspace=0.22)
    ax_mem = fig.add_subplot(gs[0, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_mem.set_facecolor(SURFACE)
    ax_time.set_facecolor(SURFACE)

    # ── Left: Peak memory ──
    y_ticks = [4, 3, 2, 1, 0]
    colors_mem = [OTHER, GREEN, TEAL, BLUE, TEAL]

    for y, (label, key), color in zip(y_ticks, configs, colors_mem):
        if key is None:
            val = DDP_PEAK_GIB
        else:
            val = d[key]["peak"]
        add_bar(ax_mem, 0, val, y, color, f"{label.split(chr(10))[0]}\n{val:.1f} GiB",
                fontsize=9.5)
        if key is not None:
            saved = DDP_PEAK_GIB - val
            ax_mem.text(val + 0.4, y, f"−{saved:.1f} GiB\n(−{saved/DDP_PEAK_GIB*100:.0f}%)",
                       fontsize=9.5, fontweight="bold", color=RED, va="center")

    ax_mem.set_xlim(0, DDP_PEAK_GIB * 1.18)
    ax_mem.set_ylim(-0.5, 4.8)
    ax_mem.set_yticks(y_ticks)
    ax_mem.set_yticklabels([c[0] for c in configs], fontsize=10, fontweight="bold", color=INK)
    ax_mem.set_xlabel("Peak GPU memory (GiB)", fontsize=10.5, color=INK)
    ax_mem.set_title("A. Peak memory before optimizer step", loc="left",
                     fontsize=12.5, fontweight="bold", color=INK, pad=12)
    ax_mem.grid(axis="x", color=GRID, linewidth=0.8)
    style_axes(ax_mem)

    # ── Right: Step time (stacked bars) ──
    y_ticks_t = [4, 3, 2, 1, 0]
    fwd_colors  = [OTHER, GREEN, TEAL, BLUE, TEAL]
    other_colors = ["#cccccc", "#d5e8d4", "#d5e8d4", "#d5e8d4", "#d5e8d4"]

    for y, (label, key), fc, oc in zip(y_ticks_t, configs, fwd_colors, other_colors):
        if key is None:
            total = DDP_TOTAL_S
            opt_t = DDP_OPT_S
            fwd_t = 0.156
            bwd_t = 0.295
            sync_t = 0.398
        else:
            total = d[key]["total"]
            fwd_t = d[key]["fwd"]
            bwd_t = d[key]["bwd"]
            sync_t = d[key]["sync"]
            opt_t = d[key]["opt"]
        # other = forward (comms hidden in fwd/bwd for FSDP)
        other_val = total - fwd_t - bwd_t - opt_t - sync_t
        add_bar(ax_time, 0, fwd_t, y, MUTED, f"fwd\n{fwd_t:.2f}s", alpha=0.6, text_color=INK, fontsize=8.5)
        add_bar(ax_time, fwd_t, bwd_t, y, fc, f"bwd\n{bwd_t:.2f}s" if bwd_t > 0.1 else "", fontsize=8.5)
        left2 = fwd_t + bwd_t
        if sync_t > 0.01:
            add_bar(ax_time, left2, sync_t, y, RED, f"sync\n{sync_t:.2f}s", alpha=0.8, fontsize=8.5)
        left3 = left2 + sync_t
        add_bar(ax_time, left3, opt_t, y, oc, f"opt\n{opt_t:.2f}s", alpha=0.75, text_color=INK, fontsize=8.5)
        ax_time.text(total + 0.04, y, f"{total:.3f}s", va="center",
                     fontsize=11, fontweight="bold", color=INK)

    ax_time.set_xlim(0, DDP_TOTAL_S * 1.25)
    ax_time.set_ylim(-0.5, 4.8)
    ax_time.set_yticks(y_ticks_t)
    ax_time.set_yticklabels([c[0] for c in configs], fontsize=10, fontweight="bold", color=INK)
    ax_time.set_xlabel("Step time (seconds)", fontsize=10.5, color=INK)
    ax_time.set_title("B. Training step time breakdown", loc="left",
                      fontsize=12.5, fontweight="bold", color=INK, pad=12)
    ax_time.grid(axis="x", color=GRID, linewidth=0.8)
    style_axes(ax_time)

    # ── Title & legend ──
    fig.suptitle("FSDP: memory savings with all-gather / reduce-scatter",
                 fontsize=17, fontweight="bold", color=INK, y=0.97)
    fig.text(0.5, 0.915, "xl model, 2-4 GPUs, batch=4",
             ha="center", fontsize=10.5, color=MUTED)
    handles = [
        Patch(facecolor=OTHER,  label="Baseline (DDP + AdamW)"),
        Patch(facecolor=GREEN, label="FSDP FP32"),
        Patch(facecolor=TEAL,  label="FSDP BF16"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.025))

    fig.subplots_adjust(left=0.12, right=0.97, top=0.84, bottom=0.16, wspace=0.25)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=220, facecolor=SURFACE, edgecolor="none")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
