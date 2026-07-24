"""Figures for scaled dot-product attention benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

plt.rcParams.update(
    {
        "font.sans-serif": ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#fafafa",
        "axes.facecolor": "#fafafa",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "grid.color": "#cccccc",
        "grid.alpha": 0.45,
    }
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

PALETTE = {
    16: "#4c72b0",
    32: "#55a868",
    64: "#c44e52",
    128: "#8172b3",
}
OOM_COLOR = "#8b0000"
LIMIT_GIB = 80.0


def make_figures(results: list[dict[str, Any]]) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "forward": _plot_timing(results, "forward_ms", "sdpa_forward_time_vs_S.png", "前向时间"),
        "backward": _plot_timing(results, "backward_ms", "sdpa_backward_time_vs_S.png", "反向时间"),
        "memory": _plot_memory(results),
        "grid": _plot_grid_summary(results),
    }


def _by_d(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in results:
        grouped.setdefault(r["d_model"], []).append(r)
    for d in grouped:
        grouped[d].sort(key=lambda x: x["seq_len"])
    return grouped


def _plot_timing(
    results: list[dict[str, Any]],
    field: str,
    filename: str,
    title_cn: str,
) -> Path:
    out = FIGURES_DIR / filename
    grouped = _by_d(results)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for d, rows in sorted(grouped.items()):
        xs_ok: list[int] = []
        ys_ok: list[float] = []
        last_ok_s: int | None = None
        first_oom_s: int | None = None

        for r in rows:
            if r.get("oom"):
                if first_oom_s is None:
                    first_oom_s = r["seq_len"]
                continue
            xs_ok.append(r["seq_len"])
            ys_ok.append(r[field])
            last_ok_s = r["seq_len"]

        color = PALETTE.get(d, "#333333")
        if xs_ok:
            ax.plot(
                xs_ok,
                ys_ok,
                "o-",
                color=color,
                linewidth=2.2,
                markersize=7,
                label=f"d={d}",
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        if last_ok_s is not None and first_oom_s is not None:
            ax.scatter(
                [first_oom_s],
                [ys_ok[-1] * 1.35 if ys_ok else 1.0],
                marker="x",
                s=120,
                color=OOM_COLOR,
                linewidths=2.5,
                zorder=5,
            )
            ax.annotate(
                "OOM",
                (first_oom_s, ys_ok[-1] * 1.35 if ys_ok else 1.0),
                textcoords="offset points",
                xytext=(6, 0),
                fontsize=9,
                color=OOM_COLOR,
                fontweight="bold",
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("序列长度 S")
    ax.set_ylabel("均值时间 (ms)")
    ax.set_title(
        f"scaled dot-product attention · {title_cn} · B=8 · 单头 · A800 80GB",
        fontsize=11,
        pad=12,
    )
    ax.legend(frameon=True, fancybox=True, shadow=False, edgecolor="#dddddd")
    ax.grid(True, which="both")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.0f}"))
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_memory(results: list[dict[str, Any]]) -> Path:
    out = FIGURES_DIR / "sdpa_memory_before_backward_vs_S.png"
    grouped = _by_d(results)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for d, rows in sorted(grouped.items()):
        xs = [r["seq_len"] for r in rows if not r.get("oom")]
        ys = [r["memory_before_backward_gib"] for r in rows if not r.get("oom")]
        if xs:
            ax.plot(
                xs,
                ys,
                "o-",
                color=PALETTE.get(d, "#333333"),
                linewidth=2.2,
                markersize=7,
                label=f"d={d}",
                markeredgecolor="white",
                markeredgewidth=0.8,
            )

    ax.axhline(LIMIT_GIB, color="#888888", linestyle="--", linewidth=1.5, label="A800 80 GiB")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("序列长度 S")
    ax.set_ylabel("backward 前 memory_allocated (GiB)")
    ax.set_title(
        "scaled dot-product attention · backward 前显存 · B=8 · 单头 · A800 80GB",
        fontsize=11,
        pad=12,
    )
    ax.legend(frameon=True, fancybox=True, edgecolor="#dddddd")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_grid_summary(results: list[dict[str, Any]]) -> Path:
    out = FIGURES_DIR / "sdpa_grid_summary.png"
    d_vals = sorted({r["d_model"] for r in results})
    s_vals = sorted({r["seq_len"] for r in results})

    lookup = {(r["d_model"], r["seq_len"]): r for r in results}
    data = np.full((len(d_vals), len(s_vals)), np.nan)
    oom_mask = np.zeros((len(d_vals), len(s_vals)), dtype=bool)

    for i, d in enumerate(d_vals):
        for j, s in enumerate(s_vals):
            r = lookup.get((d, s))
            if r is None:
                continue
            if r.get("oom"):
                oom_mask[i, j] = True
            else:
                data[i, j] = r["memory_before_backward_gib"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#eeeeee")
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=max(np.nanmax(data), 1))

    for i, d in enumerate(d_vals):
        for j, s in enumerate(s_vals):
            r = lookup.get((d, s))
            if r is None:
                continue
            if r.get("oom"):
                ax.text(j, i, "OOM", ha="center", va="center", fontsize=9, fontweight="bold", color=OOM_COLOR)
            else:
                val = r["memory_before_backward_gib"]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color="#222222")

    ax.set_xticks(range(len(s_vals)))
    ax.set_xticklabels([str(s) for s in s_vals], rotation=35, ha="right")
    ax.set_yticks(range(len(d_vals)))
    ax.set_yticklabels([f"d={d}" for d in d_vals])
    ax.set_xlabel("序列长度 S")
    ax.set_title("网格总览 · backward 前显存 (GiB) · 红格为 OOM", fontsize=11, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("GiB")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out
