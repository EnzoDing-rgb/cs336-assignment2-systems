"""Figures for gradient checkpointing sweep results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"


def _label(seg: int | None) -> str:
    return "none" if seg is None else str(seg)


def _gib(n: int | None) -> float | None:
    if n is None:
        return None
    return n / (1024**3)


def make_figures(
    results_primary: list[dict[str, Any]],
    *,
    results_appendix: list[dict[str, Any]] | None = None,
    best_segment: int | None = None,
) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["main_bars"] = _plot_main_bars(results_primary, best_segment)
    paths["successful_curve"] = _plot_successful_curve(results_primary)
    paths["neighborhood"] = _plot_neighborhood(results_primary, best_segment)

    if results_appendix:
        paths["appendix_512"] = _plot_appendix_curve(results_appendix)

    return paths


def _plot_main_bars(results: list[dict[str, Any]], best_segment: int | None) -> Path:
    out = FIGURES_DIR / "gc_peak_by_segment_2048.png"
    labels = [_label(r["segment_size"]) for r in results]
    peaks = [_gib(r.get("peak_allocated_bytes")) for r in results]
    ooms = [r.get("oom", False) for r in results]

    x = np.arange(len(labels))
    colors = []
    for r in results:
        seg = r["segment_size"]
        if r.get("oom"):
            colors.append("#c44e52")
        elif seg == best_segment:
            colors.append("#55a868")
        else:
            colors.append("#4c72b0")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(x, [p if p is not None else 0 for p in peaks], color=colors, edgecolor="#333")
    ymax = max([p for p in peaks if p is not None], default=80)
    for i, (bar, oom, peak) in enumerate(zip(bars, ooms, peaks)):
        if oom:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ymax * 0.04,
                "OOM",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#8b0000",
            )
            bar.set_hatch("//")
            bar.set_alpha(0.45)
        elif peak is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                peak + ymax * 0.02,
                f"{peak:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("segment_size (none = baseline)")
    ax.set_ylabel("peak allocated (GiB)")
    ax.set_title("xl train step peak memory · B=4 · S=2048 · NVIDIA A800 80GB")
    ax.set_ylim(0, ymax * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_successful_curve(results: list[dict[str, Any]]) -> Path:
    out = FIGURES_DIR / "gc_peak_vs_k_2048.png"
    ok = [r for r in results if r.get("segment_size") is not None and not r.get("oom")]
    ok = sorted(ok, key=lambda r: r["segment_size"])
    if not ok:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no successful k runs", ha="center", va="center")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    ks = [r["segment_size"] for r in ok]
    peaks = [_gib(r["peak_allocated_bytes"]) for r in ok]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ks, peaks, "o-", color="#4c72b0", linewidth=2, markersize=8)
    ax.set_xlabel("segment_size k")
    ax.set_ylabel("peak allocated (GiB)")
    ax.set_title("Successful runs only · S=2048")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_neighborhood(results: list[dict[str, Any]], best_segment: int | None) -> Path:
    out = FIGURES_DIR / "gc_neighborhood_best_k.png"
    if best_segment is None:
        best_segment = 1
    neighbors = sorted({max(1, best_segment - 1), best_segment, best_segment + 1})
    subset = [r for r in results if r.get("segment_size") in neighbors]
    subset = sorted(subset, key=lambda r: r["segment_size"])

    labels = [str(r["segment_size"]) for r in subset]
    peaks = [_gib(r.get("peak_allocated_bytes")) for r in subset]
    ooms = [r.get("oom", False) for r in subset]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    x = np.arange(len(labels))
    colors = ["#55a868" if r["segment_size"] == best_segment else "#4c72b0" for r in subset]
    bars = ax.bar(x, [p if p is not None else 0 for p in peaks], color=colors, edgecolor="#333")
    ymax = max([p for p in peaks if p is not None], default=80)
    for bar, oom, peak in zip(bars, ooms, peaks):
        if oom:
            ax.text(bar.get_x() + bar.get_width() / 2, ymax * 0.04, "OOM", ha="center", fontweight="bold")
            bar.set_hatch("//")
        elif peak is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, peak + ymax * 0.02, f"{peak:.1f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("segment_size k")
    ax.set_ylabel("peak allocated (GiB)")
    ax.set_title(f"Neighborhood of k*={best_segment} · S=2048")
    ax.set_ylim(0, ymax * 1.2 if ymax else 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_appendix_curve(results: list[dict[str, Any]]) -> Path:
    out = FIGURES_DIR / "gc_peak_by_segment_512.png"
    labels = [_label(r["segment_size"]) for r in results]
    peaks = [_gib(r.get("peak_allocated_bytes")) for r in results]
    ooms = [r.get("oom", False) for r in results]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#c44e52" if oom else "#8172b3" for oom in ooms]
    ax.bar(x, [p if p is not None else 0 for p in peaks], color=colors, edgecolor="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("segment_size")
    ax.set_ylabel("peak allocated (GiB)")
    ax.set_title("Appendix sweep · B=4 · S=512 · A800 80GB")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
