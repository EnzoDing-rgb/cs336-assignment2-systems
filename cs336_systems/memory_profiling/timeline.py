"""Reconstruct Active Memory Timeline from a PyTorch memory snapshot pickle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected snapshot type: {type(data)}")
    return data


def reconstruct_active_timeline(
    snapshot: dict[str, Any],
    *,
    device_index: int = 0,
    baseline_bytes: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_us relative to first event, active_bytes).

    Trace delta follows memory_viz (``alloc`` + / ``free_completed`` −).
    ``baseline_bytes`` is memory already allocated when recording started
    (weights / Adam / etc.), so the curve matches absolute device occupancy.
    """
    traces = snapshot["device_traces"][device_index]
    if not traces:
        return np.array([]), np.array([])

    t0 = traces[0]["time_us"]
    times: list[float] = [0.0]
    active: list[float] = [float(baseline_bytes)]
    cur = float(baseline_bytes)
    for ev in traces:
        action = ev["action"]
        if action == "alloc":
            cur += ev["size"]
        elif action == "free_completed":
            cur -= ev["size"]
        times.append(ev["time_us"] - t0)
        active.append(cur)
    return np.asarray(times, dtype=np.float64), np.asarray(active, dtype=np.float64)


def largest_allocs(
    snapshot: dict[str, Any],
    *,
    device_index: int = 0,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Largest ``alloc`` events with truncated Python-ish stack frames."""
    traces = snapshot["device_traces"][device_index]
    rows: list[dict[str, Any]] = []
    for ev in traces:
        if ev["action"] != "alloc":
            continue
        frames = ev.get("frames") or []
        pretty: list[str] = []
        for fr in frames:
            name = fr.get("name") or "?"
            filename = fr.get("filename") or ""
            line = fr.get("line") or 0
            if "torch::unwind" in name or "CapturedTraceback" in name:
                continue
            if filename and filename != "??":
                base = filename.rsplit("/", 1)[-1]
                pretty.append(f"{base}:{line} {name}")
            else:
                pretty.append(name)
            if len(pretty) >= 8:
                break
        rows.append(
            {
                "size_bytes": int(ev["size"]),
                "size_mib": ev["size"] / (1024**2),
                "time_us": int(ev["time_us"]),
                "frames": pretty,
            }
        )
    rows.sort(key=lambda r: r["size_bytes"], reverse=True)
    return rows[:top_k]


def plot_active_timeline(
    snapshot_path: Path,
    out_path: Path,
    *,
    title: str,
    stage_boundaries: list[dict[str, Any]] | None = None,
    baseline_bytes: int = 0,
) -> dict[str, Any]:
    """Plot Active Memory Timeline PNG; annotate stage boundaries when provided.

    ``stage_boundaries`` entries: ``{"name": str, "time_us": int}`` absolute time_us
    matching the snapshot clock (Unix microseconds).
    """
    snap = load_snapshot(snapshot_path)
    times, active = reconstruct_active_timeline(snap, baseline_bytes=baseline_bytes)
    if times.size == 0:
        raise RuntimeError(f"Empty device_traces in {snapshot_path}")

    t0_abs = int(snap["device_traces"][0][0]["time_us"])

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(times * 1e-6, active / (1024**3), color="#1f4e79", lw=1.4)
    ax.set_xlabel("时间 (s，相对录制起点)")
    ax.set_ylabel("Active memory (GiB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    peak_gib = float(active.max() / (1024**3))
    ax.axhline(peak_gib, color="#c0392b", ls="--", lw=1.0, alpha=0.7)
    ax.text(
        0.99,
        0.98,
        f"peak ≈ {peak_gib:.2f} GiB",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#c0392b",
    )

    colors = {
        "forward": "#27ae60",
        "loss": "#8e44ad",
        "backward": "#e67e22",
        "optimizer": "#2980b9",
        "end": "#7f8c8d",
    }
    if stage_boundaries:
        for b in stage_boundaries:
            name = b["name"]
            if name == "start":
                continue
            t_rel_s = (int(b["time_us"]) - t0_abs) * 1e-6
            if t_rel_s < -0.05 or t_rel_s > times[-1] * 1e-6 + 1.0:
                continue
            ax.axvline(t_rel_s, color=colors.get(name, "#333"), ls=":", lw=1.2, alpha=0.85)
            ax.text(
                t_rel_s,
                0.02,
                name,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="bottom",
                ha="right",
                fontsize=8,
                color=colors.get(name, "#333"),
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return {
        "peak_active_gib_from_trace": peak_gib,
        "n_events": int(times.size),
        "duration_s": float(times[-1] * 1e-6),
    }
