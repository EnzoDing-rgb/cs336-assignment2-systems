"""
Plot Naive DDP benchmark results → PNG figures.

Reads:
  artifacts/naive_ddp_benchmark.csv
  artifacts/naive_ddp_per_param_latency.csv

Writes:
  reports/figures/naive_ddp_step_breakdown.png    — Figure 1: stacked bar
  reports/figures/naive_ddp_batch_sweep.png       — Figure 2: batch size sweep
  reports/figures/naive_ddp_per_param_latency.png — Figure 3 (appendix): scatter
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Shared palette (matching existing report style)
# ---------------------------------------------------------------------------
BLUE      = "#2a78d6"
ORANGE    = "#eb6834"
GREEN     = "#3b9e6e"
RED       = "#d64545"
PURPLE    = "#7b5ea7"
TEAL      = "#3b9e9e"
SURFACE   = "#fcfcfb"
GRIDLINE  = "#e1e0d9"
MUTED     = "#898781"
PRIMARY_INK = "#0b0b0b"

# Segment colors for stacked bar
SEGMENT_COLORS = {
    "forward":      BLUE,
    "loss":         MUTED,
    "backward":     ORANGE,
    "gradient_sync": RED,
    "optimizer":    PURPLE,
}
SEGMENT_LABELS = {
    "forward":       "Forward",
    "loss":          "Loss",
    "backward":      "Backward",
    "gradient_sync": "Gradient Sync (all-reduce)",
    "optimizer":     "Optimizer",
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "artifacts" / "naive_ddp_benchmark.csv"
PP_CSV_PATH = REPO_ROOT / "artifacts" / "naive_ddp_per_param_latency.csv"
FIG_DIR = REPO_ROOT / "reports" / "figures"

OUT_BREAKDOWN = FIG_DIR / "naive_ddp_step_breakdown.png"
OUT_SWEEP = FIG_DIR / "naive_ddp_batch_sweep.png"
OUT_PER_PARAM = FIG_DIR / "naive_ddp_per_param_latency.png"

SEGMENTS = ["forward", "loss", "backward", "gradient_sync", "optimizer"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class Summary:
    mode: str
    batch_size: int
    n_steps: int
    means: dict[str, float]  # segment → mean seconds
    stds: dict[str, float]   # segment → std seconds
    total_mean: float
    total_std: float


def _load_summaries(csv_path: Path) -> list[Summary]:
    """Load CSV and compute per-(mode, batch_size) summaries."""
    from collections import defaultdict
    groups: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {seg: [] for seg in ["forward", "loss", "backward", "gradient_sync", "optimizer", "total"]}
    )
    with csv_path.open("r") as f:
        for row in csv.DictReader(f):
            key = (row["mode"], int(row["batch_size"]))
            groups[key]["forward"].append(float(row["forward_s"]))
            groups[key]["loss"].append(float(row["loss_s"]))
            groups[key]["backward"].append(float(row["backward_s"]))
            groups[key]["gradient_sync"].append(float(row["gradient_sync_s"]))
            groups[key]["optimizer"].append(float(row["optimizer_s"]))
            groups[key]["total"].append(float(row["total_s"]))

    summaries: list[Summary] = []
    for (mode, bs), d in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        means = {seg: float(np.mean(d[seg])) for seg in SEGMENTS}
        stds = {seg: float(np.std(d[seg])) for seg in SEGMENTS}
        summaries.append(Summary(
            mode=mode, batch_size=bs, n_steps=len(d["total"]),
            means=means, stds=stds,
            total_mean=float(np.mean(d["total"])),
            total_std=float(np.std(d["total"])),
        ))
    return summaries


def _setup_ax(ax: plt.Axes) -> None:
    """Apply consistent styling."""
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=PRIMARY_INK, labelsize=9)
    ax.grid(True, color=GRIDLINE, linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRIDLINE)


# ---------------------------------------------------------------------------
# Figure 1 — Stacked horizontal bar: step breakdown for batch=4 NaiveDDP
# ---------------------------------------------------------------------------
def plot_step_breakdown(summaries: list[Summary], out_path: Path) -> None:
    ddp_b4 = next((s for s in summaries if s.mode == "naive_ddp" and s.batch_size == 4), None)
    if ddp_b4 is None:
        print("[plot] skip step_breakdown: no naive_ddp batch=4 data")
        return

    fig, ax = plt.subplots(figsize=(11, 3.2))
    _setup_ax(ax)

    bar_order = ["forward", "loss", "backward", "gradient_sync", "optimizer"]
    left = 0.0
    y_pos = 0

    for seg in bar_order:
        val = ddp_b4.means[seg]
        pct = (val / ddp_b4.total_mean) * 100
        color = SEGMENT_COLORS[seg]
        label = SEGMENT_LABELS[seg]
        bar = ax.barh(y_pos, val, left=left, color=color, height=0.55, alpha=0.92, zorder=3)

        # Segment label + time + percentage inside bar (if wide enough)
        if pct > 4:
            text = f"{label}\n{val:.3f}s ({pct:.1f}%)"
            ax.text(left + val / 2, y_pos, text, ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold")
        left += val

    # Total label on the right
    ax.text(left + 0.02, y_pos,
            f"Total: {ddp_b4.total_mean:.3f}s",
            va="center", fontsize=10, fontweight="bold", color=PRIMARY_INK)

    ax.set_xlim(0, left * 1.15)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", color=PRIMARY_INK, fontsize=10)
    ax.set_title(
        "Naive DDP Training Step Breakdown (xl model, batch=4, 2 GPUs)",
        color=PRIMARY_INK, fontsize=12, fontweight="bold",
    )
    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=SEGMENT_COLORS[seg], alpha=0.92, label=SEGMENT_LABELS[seg])
        for seg in bar_order
    ]
    ax.legend(handles=legend_patches, loc="lower right", ncol=3,
              fontsize=8, framealpha=0.85, edgecolor=GRIDLINE)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] → {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Batch size sweep: line chart
# ---------------------------------------------------------------------------
def plot_batch_sweep(summaries: list[Summary], out_path: Path) -> None:
    single = [s for s in summaries if s.mode == "single"]
    ddp = [s for s in summaries if s.mode == "naive_ddp"]

    if not single or not ddp:
        print("[plot] skip batch_sweep: missing data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    _setup_ax(ax1)
    _setup_ax(ax2)

    # --- Left: Absolute times ---
    bs_single = [s.batch_size for s in single]
    bs_ddp = [s.batch_size for s in ddp]

    ax1.plot(bs_single, [s.total_mean for s in single],
             "o-", color=BLUE, linewidth=2, markersize=7, label="Single GPU (no DDP)")
    ax1.fill_between(bs_single,
                     [s.total_mean - s.total_std for s in single],
                     [s.total_mean + s.total_std for s in single],
                     color=BLUE, alpha=0.12)

    ax1.plot(bs_ddp, [s.total_mean for s in ddp],
             "s-", color=RED, linewidth=2, markersize=7, label="2-GPU Naive DDP")
    ax1.fill_between(bs_ddp,
                     [s.total_mean - s.total_std for s in ddp],
                     [s.total_mean + s.total_std for s in ddp],
                     color=RED, alpha=0.12)

    # Gradient sync time (filled area)
    sync_means = [s.means["gradient_sync"] for s in ddp]
    ax1.fill_between(bs_ddp, 0, sync_means, color=RED, alpha=0.18,
                     label="Communication (gradient sync)")

    ax1.set_xlabel("Total Batch Size", color=PRIMARY_INK, fontsize=10)
    ax1.set_ylabel("Time per Step (s)", color=PRIMARY_INK, fontsize=10)
    ax1.set_title("Absolute Step Time", color=PRIMARY_INK, fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, framealpha=0.85, edgecolor=GRIDLINE)
    ax1.set_xticks(bs_ddp)
    ax1.set_xlim(min(bs_ddp) - 1, max(bs_ddp) + 1)

    # --- Right: Communication proportion ---
    comm_pcts = [
        (s.means["gradient_sync"] / s.total_mean * 100)
        if s.total_mean > 0 else 0.0
        for s in ddp
    ]
    ax2.bar(bs_ddp, comm_pcts, color=RED, alpha=0.75, width=2.5, zorder=3)
    # Annotate percentages
    for bs, pct in zip(bs_ddp, comm_pcts):
        ax2.text(bs, pct + 1.0, f"{pct:.1f}%", ha="center", fontsize=9,
                 fontweight="bold", color=PRIMARY_INK)

    # Compute-per-GPU time (without comm) for reference
    compute_ddp = [
        s.means["forward"] + s.means["loss"] + s.means["backward"] + s.means["optimizer"]
        for s in ddp
    ]
    # per-sample efficiency: single-GPU time per sample vs DDP time per sample
    perf_single = [s.total_mean / (s.batch_size) for s in single if s.batch_size in bs_ddp]
    perf_ddp = [s.total_mean / (s.batch_size) for s in ddp]

    ax2.set_xlabel("Total Batch Size", color=PRIMARY_INK, fontsize=10)
    ax2.set_ylabel("Communication Proportion (%)", color=RED, fontsize=10)
    ax2.set_title("Communication Overhead", color=PRIMARY_INK, fontsize=11, fontweight="bold")
    ax2.set_xticks(bs_ddp)
    ax2.set_xlim(min(bs_ddp) - 1, max(bs_ddp) + 1)
    ax2.set_ylim(0, max(comm_pcts) * 1.25 if comm_pcts else 100)

    fig.suptitle(
        "Naive DDP Benchmark — Batch Size Sweep (xl model, 2 GPUs)",
        color=PRIMARY_INK, fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] → {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 (appendix) — Per-parameter all-reduce latency scatter
# ---------------------------------------------------------------------------
def plot_per_param_latency(pp_csv_path: Path, out_path: Path) -> None:
    if not pp_csv_path.exists():
        print(f"[plot] skip per_param: {pp_csv_path} not found")
        return

    names, numels, bytess, lats = [], [], [], []
    with pp_csv_path.open("r") as f:
        for row in csv.DictReader(f):
            numels.append(int(row["numel"]))
            bytess.append(int(row["bytes"]))
            lats.append(float(row["latency_s"]) * 1e3)  # convert to ms
            names.append(row["param_name"])

    if not numels:
        print("[plot] skip per_param: no data rows")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _setup_ax(ax)

    # Categorize by tensor type for coloring
    colors = []
    for name in names:
        if "embedding" in name or "lm_head" in name or "token_embedding" in name:
            colors.append(BLUE)
        elif "rmsnorm" in name.lower() or "norm" in name.lower() or "ln_" in name:
            colors.append(ORANGE)
        elif "ff" in name or "fc" in name:
            colors.append(GREEN)
        elif "attention" in name.lower() or "attn" in name.lower() or "wq" in name or "wk" in name or "wv" in name or "wo" in name:
            colors.append(PURPLE)
        else:
            colors.append(MUTED)

    ax.scatter(bytess, lats, c=colors, alpha=0.55, s=18, edgecolors="none", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tensor Size (bytes)", color=PRIMARY_INK, fontsize=10)
    ax.set_ylabel("Single all_reduce Latency (ms)", color=PRIMARY_INK, fontsize=10)
    ax.set_title(
        "Per-Parameter all_reduce Latency vs. Tensor Size (xl model, 2 GPUs)",
        color=PRIMARY_INK, fontsize=12, fontweight="bold",
    )

    # Annotate extremes
    if bytess and lats:
        min_idx = np.argmin(bytess)
        max_idx = np.argmax(bytess)
        ax.annotate(
            f"RMSNorm\n{bytess[min_idx]:,} B\n{lats[min_idx]:.3f} ms",
            (bytess[min_idx], lats[min_idx]),
            textcoords="offset points", xytext=(10, -15),
            fontsize=7, color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, alpha=0.6),
        )
        ax.annotate(
            f"Embedding\n{bytess[max_idx]:,} B\n{lats[max_idx]:.1f} ms",
            (bytess[max_idx], lats[max_idx]),
            textcoords="offset points", xytext=(-80, 10),
            fontsize=7, color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE, alpha=0.6),
        )

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor=BLUE, alpha=0.7, label="Embedding / LM head"),
        Patch(facecolor=ORANGE, alpha=0.7, label="RMSNorm"),
        Patch(facecolor=GREEN, alpha=0.7, label="FFN weights"),
        Patch(facecolor=PURPLE, alpha=0.7, label="Attention weights"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, framealpha=0.85, edgecolor=GRIDLINE)

    # Bandwidth reference lines
    # Theoretical bandwidths: 100 GB/s, 10 GB/s, 1 GB/s
    x_ref = np.logspace(3, 8.5, 100)  # 1KB to ~300 MB
    for bw_gbps, ls, lbl in [(50, "--", "50 GB/s (PCIe 5.0 x16 peak)"), (12, "-.", "12 GB/s (typical NCCL ring)")]:
        # latency = bytes / bandwidth
        y_ref = (x_ref / (bw_gbps * 1e9)) * 1e3  # → ms
        ax.plot(x_ref, y_ref, ls, color=MUTED, alpha=0.5, linewidth=1, label=lbl)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plot] → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summaries = _load_summaries(CSV_PATH)
    if not summaries:
        print("[plot] ERROR: no data loaded from CSV. Run benchmark_naive_ddp.py first.")
        return

    print(f"[plot] loaded {len(summaries)} summary groups:")
    for s in summaries:
        print(f"  {s.mode:>10}  batch={s.batch_size:>2}  "
              f"n={s.n_steps}  total={s.total_mean:.3f}s ± {s.total_std:.3f}s")

    plot_step_breakdown(summaries, OUT_BREAKDOWN)
    plot_batch_sweep(summaries, OUT_SWEEP)
    plot_per_param_latency(PP_CSV_PATH, OUT_PER_PARAM)

    print("\n✓ All figures written to reports/figures/")


if __name__ == "__main__":
    main()
