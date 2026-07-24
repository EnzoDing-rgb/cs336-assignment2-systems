"""Nsight Systems profiling for Assignment 2 `nsys_profile` parts (a) and (b).

One profiled run per (model_size, context_length):
  outer NVTX "forward_backward" wraps forward → loss → backward;
  inner NVTX "forward" wraps model(x) + cuda synchronize.

Python baseline (no nsys) times the same forward region for part (a).

Headless only: nsys profile → nsys stats (nvtxsum, nvtxkernsum) → plots/report.
No Nsight GUI.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
import timeit
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.e2e_timing.e2e import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_VOCAB_SIZE,
    DEFAULT_WARMUP,
    MODEL_SIZE_PRESETS,
    BenchmarkConfig,
    _format_seconds,
    _mean_std,
    _sync,
    build_model,
    make_batch,
)

# Prefer a CJK-capable font so share-pie labels render correctly.
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "nsys_profile"
REPORT_PATH = REPO_ROOT / "reports" / "nsys-profile.md"
FIGURE_A = REPO_ROOT / "reports" / "figures" / "nsys_a_forward_python_vs_nsys.png"
FIGURE_B = REPO_ROOT / "reports" / "figures" / "nsys_b_top_kernels.png"
FIGURE_B_SHARE_OVERVIEW = REPO_ROOT / "reports" / "figures" / "nsys_b_share_overview.png"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
SUITE_MANIFEST = ARTIFACTS_ROOT / "suite_manifest.json"

NSYS_SIZES: tuple[str, ...] = ("medium", "xl")
NSYS_CONTEXTS: tuple[int, ...] = (256, 512, 1024)

NVTX_FORWARD = "forward"
NVTX_STEP = "forward_backward"
NVTX_WARMUP = "warmup"

TOP_K = 5


def _rel(path: Path) -> str:
    """Path relative to repo root when possible (for reports/manifests)."""
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class CellResult:
    model_size: str
    context_length: int
    python_forward_mean_s: float
    python_forward_std_s: float
    nsys_forward_mean_s: float | None
    nsys_forward_instances: int | None
    top_kernel_forward: str | None
    top_kernel_forward_calls_per_step: float | None
    top_kernel_forward_total_s: float | None
    top_kernel_step: str | None
    top_kernel_step_total_s: float | None
    same_top_kernel: bool | None
    top5_forward: list[dict[str, Any]]
    top5_step: list[dict[str, Any]]
    oom: bool = False
    artifact_dir: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CellResult:
        return cls(**d)


def _cfg(size: str, context: int, warmup: int = DEFAULT_WARMUP, steps: int = DEFAULT_STEPS) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_size=size,
        mode="forward_backward",
        vocab_size=DEFAULT_VOCAB_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        context_length=context,
        warmup=warmup,
        steps=steps,
        seed=DEFAULT_SEED,
        device="cuda",
    )


def _run_step_with_nvtx(
    model: torch.nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    label_outer: str,
    label_forward: str,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with nvtx.range(label_outer):
        with nvtx.range(label_forward):
            logits = model(x)
            _sync(device)
        loss = cross_entropy(logits, y)
        loss.backward()
        _sync(device)


def run_python_forward_baseline(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Time forward-only in a *subprocess* so GPU memory is fully released afterwards.

    Running in-process leaves CUDA caching allocator state that can OOM the subsequent
    nsys child when profiling large models (two ~40GiB residents).
    """
    cmd = [
        sys.executable,
        "-m",
        "cs336_systems.nsys_profile.ab",
        "python-baseline",
        "--model-size",
        cfg.model_size,
        "--context-length",
        str(cfg.context_length),
        "--warmup",
        str(cfg.warmup),
        "--steps",
        str(cfg.steps),
        "--seed",
        str(cfg.seed),
    ]
    print(f"[python-baseline] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "") + "\n" + (proc.stdout or "")
        if "out of memory" in err.lower():
            raise torch.cuda.OutOfMemoryError(err[-2000:])
        raise RuntimeError(f"python-baseline failed ({proc.returncode}): {err[-2000:]}")
    # last JSON object in stdout
    text = proc.stdout.strip()
    start = text.rfind("{")
    if start < 0:
        raise RuntimeError(f"no JSON from python-baseline: {text[-500:]}")
    return json.loads(text[start:])


def _run_python_forward_baseline_inplace(cfg: BenchmarkConfig) -> dict[str, Any]:
    """In-process forward timing (used by the `python-baseline` CLI entry)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.empty_cache()

    model = build_model(cfg, device)
    model.train()
    x, _y = make_batch(cfg, device)

    for _ in range(cfg.warmup):
        _ = model(x)
        _sync(device)

    times: list[float] = []
    for _ in range(cfg.steps):
        _sync(device)
        t0 = timeit.default_timer()
        _ = model(x)
        _sync(device)
        times.append(timeit.default_timer() - t0)

    mean_s, std_s = _mean_std(times)
    del model
    del x
    del _y
    torch.cuda.empty_cache()
    return {
        "model_size": cfg.model_size,
        "context_length": cfg.context_length,
        "forward_times_s": times,
        "forward_mean_s": mean_s,
        "forward_std_s": std_s,
    }


def run_nvtx_workload(cfg: BenchmarkConfig) -> None:
    """Workload intended to be launched under `nsys profile`."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.empty_cache()

    print(
        f"[workload] size={cfg.model_size} ctx={cfg.context_length} "
        f"warmup={cfg.warmup} steps={cfg.steps}",
        flush=True,
    )
    model = build_model(cfg, device)
    model.train()
    optimizer = AdamW(model.parameters())
    x, y = make_batch(cfg, device)

    for i in range(cfg.warmup):
        with nvtx.range(NVTX_WARMUP):
            _run_step_with_nvtx(
                model,
                optimizer,
                x,
                y,
                device,
                label_outer=f"{NVTX_WARMUP}_step",
                label_forward=f"{NVTX_WARMUP}_forward",
            )
        print(f"[workload] warmup {i + 1}/{cfg.warmup}", flush=True)

    for i in range(cfg.steps):
        _run_step_with_nvtx(
            model,
            optimizer,
            x,
            y,
            device,
            label_outer=NVTX_STEP,
            label_forward=NVTX_FORWARD,
        )
        print(f"[workload] measure {i + 1}/{cfg.steps}", flush=True)

    print("[workload] done", flush=True)


def _nsys_bin() -> str:
    """Prefer a CUDA-12-capable Nsight Systems (stock 2022.4 on this box drops kernels)."""
    import os

    candidates = [
        os.environ.get("NSYS_BIN"),
        str(REPO_ROOT / "tools" / "nsight-systems-2024.2.3" / "bin" / "nsys"),
        "/tmp/nsys_extract/opt/nvidia/nsight-systems/2024.2.3/bin/nsys",
        shutil.which("nsys"),
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    raise RuntimeError(
        "No usable nsys found. Extract Nsight Systems ≥2024.2 under tools/ "
        "or set NSYS_BIN."
    )


def profile_with_nsys(cfg: BenchmarkConfig, out_prefix: Path) -> Path:
    """Run workload under nsys; return path to .nsys-rep."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rep = Path(str(out_prefix) + ".nsys-rep")
    if rep.exists():
        rep.unlink()

    # Do not pass a bare `--` before the app: Nsight 2022.4 treats later -flags as its own.
    # Do not use --capture-range=nvtx here: warmup is filtered later by NVTX range name.
    cmd = [
        _nsys_bin(),
        "profile",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda,nvtx",
        "--force-overwrite=true",
        "-o",
        str(out_prefix),
        sys.executable,
        "-m",
        "cs336_systems.nsys_profile.ab",
        "workload",
        "--model-size",
        cfg.model_size,
        "--context-length",
        str(cfg.context_length),
        "--warmup",
        str(cfg.warmup),
        "--steps",
        str(cfg.steps),
        "--seed",
        str(cfg.seed),
    ]
    print(f"[nsys] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    # Always try to free any leftover CUDA state in this process after the child exits.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if proc.returncode != 0:
        # nsys often returns 1 on workload OOM; surface as OOM when stderr/log hints so.
        raise RuntimeError(f"nsys profile failed with exit code {proc.returncode}")
    if not rep.exists():
        raise RuntimeError(f"expected report missing: {rep}")
    return rep


def export_stats(rep_path: Path, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "stats"
    # Report names for Nsight Systems 2024.x (older aliases: nvtxsum / nvtxkernsum).
    cmd = [
        _nsys_bin(),
        "stats",
        "--force-export=true",
        "--force-overwrite=true",
        "--format=csv",
        f"--output={base}",
        "--report=nvtx_pushpop_sum",
        "--report=nvtx_kern_sum",
        str(rep_path),
    ]
    print(f"[nsys] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"nsys stats failed with exit code {proc.returncode}")

    found: dict[str, Path] = {}
    for path in out_dir.glob("stats*.csv"):
        name = path.name.lower()
        if "nvtx_pushpop_sum" in name or (name.endswith("nvtxsum.csv") and "kern" not in name):
            found["nvtxsum"] = path
        elif "nvtx_kern_sum" in name or "nvtxkernsum" in name:
            found["nvtxkernsum"] = path
    if "nvtxsum" not in found or "nvtxkernsum" not in found:
        raise RuntimeError(f"could not find stats CSVs in {out_dir}; have={[p.name for p in out_dir.iterdir()]}")
    return found


def _read_csv(path: Path) -> list[dict[str, str]]:
    # nsys CSV often has a title line before the header
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_idx = None
    for i, line in enumerate(text):
        if "Total Time" in line or line.startswith("Time,"):
            header_idx = i
            break
        if "NVTX Range" in line and "Kernel Name" in line:
            header_idx = i
            break
        if line.startswith("Range,") or ",Range" in line[:40]:
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(text):
            if "," in line and not line.startswith("#"):
                header_idx = i
                break
    if header_idx is None:
        return []
    return list(csv.DictReader(text[header_idx:]))


def _row_get(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        # tolerate "Total Time (ns)" style headers
        for actual in row:
            if actual.lower() == key.lower() or actual.lower().startswith(key.lower() + " "):
                if row[actual] not in (None, ""):
                    return row[actual]
    return ""


def _ns_to_s(value: str | float | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value) / 1e9


def parse_nvtx_forward_mean(nvtxsum_csv: Path) -> tuple[float, int]:
    rows = _read_csv(nvtxsum_csv)
    for row in rows:
        range_name = _row_get(row, "Range", "NVTX Range").strip()
        if range_name == NVTX_FORWARD:
            avg = _row_get(row, "Avg", "Average")
            inst = int(float(_row_get(row, "Instances", "NVTX Inst") or 0))
            return _ns_to_s(avg), inst
    raise RuntimeError(f"NVTX range {NVTX_FORWARD!r} not found in {nvtxsum_csv}")


def parse_top_kernels(nvtxkernsum_csv: Path, range_name: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    rows = _read_csv(nvtxkernsum_csv)
    matched: list[dict[str, Any]] = []
    for row in rows:
        rname = _row_get(row, "NVTX Range", "Range").strip()
        if rname != range_name:
            continue
        kname = _row_get(row, "Kernel Name", "Name").strip()
        if not kname:
            continue
        total_s = _ns_to_s(_row_get(row, "Total Time"))
        kern_inst = float(_row_get(row, "Kern Inst", "Instances") or 0)
        matched.append(
            {
                "kernel": kname,
                "total_s": total_s,
                "kern_inst": kern_inst,
                "nvtx_inst": float(_row_get(row, "NVTX Inst") or 0),
            }
        )
    # Aggregate duplicate kernel names (different grid/block can appear as separate rows in some exports)
    by_name: dict[str, dict[str, Any]] = {}
    for item in matched:
        cur = by_name.get(item["kernel"])
        if cur is None:
            by_name[item["kernel"]] = dict(item)
        else:
            cur["total_s"] += item["total_s"]
            cur["kern_inst"] += item["kern_inst"]
    ranked = sorted(by_name.values(), key=lambda d: d["total_s"], reverse=True)
    return ranked[:top_k]


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg or isinstance(err, torch.cuda.OutOfMemoryError)


def run_cell(size: str, context: int, cell_dir: Path) -> CellResult:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(size, context)
    notes = ""

    # Python baseline
    try:
        print(f"[cell] python baseline {size} ctx={context}", flush=True)
        py = run_python_forward_baseline(cfg)
        (cell_dir / "python_forward.json").write_text(json.dumps(py, indent=2), encoding="utf-8")
    except Exception as err:  # noqa: BLE001
        if _is_oom(err):
            torch.cuda.empty_cache()
            return CellResult(
                model_size=size,
                context_length=context,
                python_forward_mean_s=float("nan"),
                python_forward_std_s=float("nan"),
                nsys_forward_mean_s=None,
                nsys_forward_instances=None,
                top_kernel_forward=None,
                top_kernel_forward_calls_per_step=None,
                top_kernel_forward_total_s=None,
                top_kernel_step=None,
                top_kernel_step_total_s=None,
                same_top_kernel=None,
                top5_forward=[],
                top5_step=[],
                oom=True,
                artifact_dir=_rel(cell_dir),
                notes=f"OOM during python baseline: {err}",
            )
        raise

    # nsys profile
    try:
        print(f"[cell] nsys profile {size} ctx={context}", flush=True)
        rep = profile_with_nsys(cfg, cell_dir / "profile")
        stats_paths = export_stats(rep, cell_dir)
        fwd_mean, fwd_inst = parse_nvtx_forward_mean(stats_paths["nvtxsum"])
        top_fwd = parse_top_kernels(stats_paths["nvtxkernsum"], NVTX_FORWARD)
        top_step = parse_top_kernels(stats_paths["nvtxkernsum"], NVTX_STEP)
    except Exception as err:  # noqa: BLE001
        if _is_oom(err):
            torch.cuda.empty_cache()
            return CellResult(
                model_size=size,
                context_length=context,
                python_forward_mean_s=py["forward_mean_s"],
                python_forward_std_s=py["forward_std_s"],
                nsys_forward_mean_s=None,
                nsys_forward_instances=None,
                top_kernel_forward=None,
                top_kernel_forward_calls_per_step=None,
                top_kernel_forward_total_s=None,
                top_kernel_step=None,
                top_kernel_step_total_s=None,
                same_top_kernel=None,
                top5_forward=[],
                top5_step=[],
                oom=True,
                artifact_dir=_rel(cell_dir),
                notes=f"OOM during nsys/workload: {err}",
            )
        raise

    best_fwd = top_fwd[0] if top_fwd else None
    best_step = top_step[0] if top_step else None
    calls = None
    if best_fwd is not None and cfg.steps > 0:
        calls = best_fwd["kern_inst"] / cfg.steps

    same = None
    if best_fwd is not None and best_step is not None:
        same = best_fwd["kernel"] == best_step["kernel"]

    result = CellResult(
        model_size=size,
        context_length=context,
        python_forward_mean_s=py["forward_mean_s"],
        python_forward_std_s=py["forward_std_s"],
        nsys_forward_mean_s=fwd_mean,
        nsys_forward_instances=fwd_inst,
        top_kernel_forward=None if best_fwd is None else best_fwd["kernel"],
        top_kernel_forward_calls_per_step=calls,
        top_kernel_forward_total_s=None if best_fwd is None else best_fwd["total_s"],
        top_kernel_step=None if best_step is None else best_step["kernel"],
        top_kernel_step_total_s=None if best_step is None else best_step["total_s"],
        same_top_kernel=same,
        top5_forward=top_fwd,
        top5_step=top_step,
        oom=False,
        artifact_dir=_rel(cell_dir),
        notes=notes,
    )
    (cell_dir / "cell_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    torch.cuda.empty_cache()
    return result


def plot_part_a(cells: list[CellResult], out_path: Path) -> None:
    ok = [c for c in cells if not c.oom and c.nsys_forward_mean_s is not None]
    sizes = [s for s in NSYS_SIZES if any(c.model_size == s for c in ok)]
    if not sizes:
        print(f"[plot] skip {out_path.name}: no data", flush=True)
        return
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 4.2), sharey=False)
    if len(sizes) == 1:
        axes = [axes]
    width = 0.35
    for ax, size in zip(axes, sizes):
        ctxs = sorted({c.context_length for c in ok if c.model_size == size})
        xs = list(range(len(ctxs)))
        py = []
        ns = []
        for ctx in ctxs:
            cell = next(c for c in ok if c.model_size == size and c.context_length == ctx)
            py.append(cell.python_forward_mean_s * 1e3)
            ns.append((cell.nsys_forward_mean_s or float("nan")) * 1e3)
        ax.bar([x - width / 2 for x in xs], py, width, label="Python timeit")
        ax.bar([x + width / 2 for x in xs], ns, width, label="nsys NVTX forward")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(c) for c in ctxs])
        ax.set_xlabel("context length")
        ax.set_title(size)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Mean forward time (ms)")
    axes[0].legend()
    fig.suptitle("Part (a): Python timer vs nsys NVTX forward mean", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _short_kernel_label(name: str) -> str:
    lower = name.lower()
    if "sgemm" in lower or "gemm" in lower:
        return name if len(name) <= 40 else name[:37] + "..."
    if "MulFunctor" in name and "vectorized_elementwise" in name:
        return "逐元素乘法（向量化）"
    if "MulFunctor" in name and "elementwise" in name:
        return "逐元素乘法"
    if "BUnaryFunctor" in name:
        return "一元逐元素（向量化）"
    if "CUDAFunctor_add" in name or "CUDAFunctorOnSelf_add" in name:
        return "逐元素加法"
    if "where_kernel" in name or "::<unnamed>::whe" in name:
        return "where / 掩码选择"
    if "softmax" in lower:
        return "softmax 相关"
    if "exp_kernel" in name:
        return "exp"
    if "sigmoid_kernel" in name:
        return "sigmoid"
    if "MaxOps" in name:
        return "reduce max"
    if "MeanOps" in name:
        return "reduce mean"
    if "func_wrapper" in name and "reduce_kernel" in name:
        return "reduce（求和类）"
    if "CatArray" in name:
        return "concat"
    if "direct_copy" in name or "copy_kernel" in name:
        return "拷贝"
    if name.startswith("void at::native::"):
        # Keep a distinctive fragment so many ATen helpers don't collapse to one label.
        core = name.removeprefix("void at::native::").split("<", 1)[0]
        core = core.split("(")[0]
        return core[:36] if len(core) <= 36 else core[:33] + "..."
    return name if len(name) <= 36 else name[:33] + "..."


def _is_gemm(name: str) -> bool:
    n = name.lower()
    return "gemm" in n or "sgemm" in n or "cutlass" in n and "gemm" in n


def _is_elementwise(name: str) -> bool:
    return "elementwise" in name or "MulFunctor" in name or "BinaryFunctor" in name


def load_range_kernel_totals(kern_csv: Path, range_name: str) -> list[tuple[str, float]]:
    """Return [(kernel_name, total_seconds), ...] aggregated and sorted desc."""
    rows = _read_csv(kern_csv)
    agg: dict[str, float] = {}
    for row in rows:
        rname = _row_get(row, "NVTX Range", "Range").strip()
        if rname != range_name:
            continue
        kname = _row_get(row, "Kernel Name", "Name").strip()
        if not kname:
            continue
        agg[kname] = agg.get(kname, 0.0) + _ns_to_s(_row_get(row, "Total Time"))
    return sorted(agg.items(), key=lambda kv: kv[1], reverse=True)


def share_breakdown(ranked: list[tuple[str, float]]) -> dict[str, Any]:
    """Top1 / top2 / top3-5 / rest as fractions of *all* CUDA kernel time in the range."""
    total = sum(t for _, t in ranked)
    if total <= 0:
        return {
            "total_s": 0.0,
            "pct_rank1": 0.0,
            "pct_rank2": 0.0,
            "pct_rank345": 0.0,
            "pct_rest": 0.0,
            "pct_gemm": 0.0,
            "pct_elementwise": 0.0,
            "rank1_name": None,
            "rank2_name": None,
            "ranked": [],
        }
    t1 = ranked[0][1] if len(ranked) > 0 else 0.0
    t2 = ranked[1][1] if len(ranked) > 1 else 0.0
    t345 = sum(t for _, t in ranked[2:5])
    rest = total - t1 - t2 - t345
    gemm = sum(t for n, t in ranked if _is_gemm(n))
    elem = sum(t for n, t in ranked if _is_elementwise(n))
    return {
        "total_s": total,
        "pct_rank1": 100.0 * t1 / total,
        "pct_rank2": 100.0 * t2 / total,
        "pct_rank345": 100.0 * t345 / total,
        "pct_rest": 100.0 * max(rest, 0.0) / total,
        "pct_gemm": 100.0 * gemm / total,
        "pct_elementwise": 100.0 * elem / total,
        "rank1_name": ranked[0][0] if ranked else None,
        "rank2_name": ranked[1][0] if len(ranked) > 1 else None,
        "ranked": [{"kernel": n, "total_s": t, "pct": 100.0 * t / total} for n, t in ranked],
    }


def _cell_kern_csv(cell: CellResult) -> Path | None:
    if not cell.artifact_dir:
        return None
    base = Path(cell.artifact_dir)
    if not base.is_absolute():
        base = REPO_ROOT / base
    matches = sorted(base.glob("stats*nvtx_kern_sum.csv"))
    return matches[-1] if matches else None


def plot_top5_bar_figure(
    items: list[dict[str, Any]],
    title: str,
    out_path: Path,
) -> None:
    """Single-panel horizontal bars: top-5 kernels by cumulative Absolute Time (ms)."""
    names = [_short_kernel_label(it["kernel"]) for it in items][::-1]
    vals = [it["total_s"] * 1e3 for it in items][::-1]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.barh(names, vals, color="#4c78a8")
    ax.set_xlabel("累计内核时间 (ms)")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_part_b_top5(cells: list[CellResult], out_path: Path) -> list[str]:
    """Restore absolute-time top-5 bars: multi-panel overview + ~10 per-cell PNGs."""
    ok = [c for c in cells if not c.oom and c.top5_forward and c.top5_step]
    top5_rels: list[str] = []
    if not ok:
        print(f"[plot] skip part-b top5: no data", flush=True)
        return top5_rels

    sizes = [s for s in NSYS_SIZES if any(c.model_size == s for c in ok)]
    ctxs = [c for c in NSYS_CONTEXTS if any(cell.context_length == c for cell in ok)]
    n_rows = len(sizes)
    n_cols = len(ctxs) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.4 * n_rows), squeeze=False)

    for r, size in enumerate(sizes):
        for i, ctx in enumerate(ctxs):
            cell = next((c for c in ok if c.model_size == size and c.context_length == ctx), None)
            ax_l = axes[r][2 * i]
            ax_r = axes[r][2 * i + 1]
            if cell is None:
                ax_l.set_visible(False)
                ax_r.set_visible(False)
                continue
            for ax, items, title in (
                (ax_l, cell.top5_forward, "forward only"),
                (ax_r, cell.top5_step, "forward+backward"),
            ):
                names = [_short_kernel_label(it["kernel"]) for it in items][::-1]
                vals = [it["total_s"] * 1e3 for it in items][::-1]
                ax.barh(names, vals, color="#4c78a8")
                ax.set_title(f"{size} ctx={ctx}\n{title}", fontsize=9)
                ax.set_xlabel("Cumulative kernel time (ms)", fontsize=8)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Part (b): top-5 CUDA kernels by cumulative time (nsys nvtxkernsum)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {out_path.name} (top-5 absolute-time multi-panel)", flush=True)

    for cell in ok:
        for items, tag in ((cell.top5_forward, "forward"), (cell.top5_step, "step")):
            fname = f"nsys_b_top5_{cell.model_size}_ctx{cell.context_length}_{tag}.png"
            out = FIGURES_DIR / fname
            title = f"{cell.model_size} · ctx={cell.context_length} · {tag}\n前 5 名内核累计时间 (ms)"
            plot_top5_bar_figure(items, title=title, out_path=out)
            top5_rels.append(f"figures/{fname}")
            print(f"[plot] wrote {fname}", flush=True)

    return top5_rels


def plot_share_figure(
    breakdown: dict[str, Any],
    *,
    title: str,
    out_path: Path,
) -> None:
    """Pie: rank1 / rank2 / ranks3-5 / other, labeled with percent of all CUDA time."""
    sizes = [
        breakdown["pct_rank1"],
        breakdown["pct_rank2"],
        breakdown["pct_rank345"],
        breakdown["pct_rest"],
    ]
    labels = [
        f"第1名\n{breakdown['pct_rank1']:.1f}%",
        f"第2名\n{breakdown['pct_rank2']:.1f}%",
        f"第3–5名合计\n{breakdown['pct_rank345']:.1f}%",
        f"其余内核\n{breakdown['pct_rest']:.1f}%",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#c7c7c7"]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    wedges, _ = ax.pie(
        sizes,
        colors=colors,
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9)
    r1 = _short_kernel_label(breakdown["rank1_name"] or "")
    r2 = _short_kernel_label(breakdown["rank2_name"] or "")
    ax.set_title(
        f"{title}\n第1={r1}\n第2={r2}\n"
        f"(占比=该范围内全部 CUDA 内核累计时间)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_part_b(
    cells: list[CellResult],
    out_path: Path,
    share_overview_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Write top-5 absolute-time bars (kept) + share pies. Returns (top5_rels, share_rels)."""
    top5_rels = plot_part_b_top5(cells, out_path)

    ok = [c for c in cells if not c.oom and c.top5_forward and c.top5_step]
    share_rels: list[str] = []
    if not ok:
        print(f"[plot] skip part-b shares: no data", flush=True)
        return top5_rels, share_rels

    for cell in ok:
        csv_path = _cell_kern_csv(cell)
        if csv_path is None:
            continue
        for range_name, tag in ((NVTX_FORWARD, "forward"), (NVTX_STEP, "step")):
            ranked = load_range_kernel_totals(csv_path, range_name)
            if not ranked:
                continue
            bd = share_breakdown(ranked)
            fname = f"nsys_b_share_{cell.model_size}_ctx{cell.context_length}_{tag}.png"
            out = FIGURES_DIR / fname
            title = f"{cell.model_size} · ctx={cell.context_length} · {tag}"
            plot_share_figure(bd, title=title, out_path=out)
            share_rels.append(f"figures/{fname}")
            print(
                f"[plot] wrote {fname}  #1={bd['pct_rank1']:.1f}% "
                f"#2={bd['pct_rank2']:.1f}% #3-5={bd['pct_rank345']:.1f}%",
                flush=True,
            )

    overview = share_overview_path or FIGURE_B_SHARE_OVERVIEW
    if share_rels:
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.55 * len(ok) + 1.5)))
        ylabels = []
        r1s, r2s, r345s = [], [], []
        for cell in ok:
            csv_path = _cell_kern_csv(cell)
            if csv_path is None:
                continue
            bd = share_breakdown(load_range_kernel_totals(csv_path, NVTX_FORWARD))
            ylabels.append(f"{cell.model_size}@ctx{cell.context_length}")
            r1s.append(bd["pct_rank1"])
            r2s.append(bd["pct_rank2"])
            r345s.append(bd["pct_rank345"])
        ys = list(range(len(ylabels)))
        ax.barh(ys, r1s, label="第1名 %", color="#1f77b4")
        ax.barh(ys, r2s, left=r1s, label="第2名 %", color="#ff7f0e")
        left2 = [a + b for a, b in zip(r1s, r2s)]
        ax.barh(ys, r345s, left=left2, label="第3–5名合计 %", color="#2ca02c")
        ax.set_yticks(ys)
        ax.set_yticklabels(ylabels)
        ax.set_xlabel("占该范围全部 CUDA 内核时间的百分比")
        ax.set_title("Part (b) 占比总览：向前传播中 top1 / top2 / top3–5")
        ax.set_xlim(0, 100)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        overview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(overview, dpi=160)
        plt.close(fig)
        print(f"[plot] wrote {overview.name} (share overview)", flush=True)

    return top5_rels, share_rels


def md_img(src: str, *, alt: str = "", width: int = 480) -> str:
    """Centered HTML image so the report doesn't stretch figures full-width."""
    a = alt or Path(src).stem
    return (
        f'<p align="center">\n'
        f'  <img src="{src}" alt="{a}" width="{width}" />\n'
        f"</p>"
    )


def write_report(
    cells: list[CellResult],
    path: Path,
    top5_figure_rels: list[str] | None = None,
    share_figure_rels: list[str] | None = None,
) -> None:
    # Markdown image links MUST stay repo-relative to the report file (never absolute).
    figure_a_rel = "figures/nsys_a_forward_python_vs_nsys.png"
    figure_b_rel = "figures/nsys_b_top_kernels.png"
    figure_b_share_overview_rel = "figures/nsys_b_share_overview.png"
    top5_figure_rels = top5_figure_rels or []
    share_figure_rels = share_figure_rels or []

    # Precompute forward-range shares for narrative (focus medium@512).
    focus_share: dict[str, Any] | None = None
    focus = next((c for c in cells if c.model_size == "medium" and c.context_length == 512 and not c.oom), None)
    if focus is not None:
        csv_path = _cell_kern_csv(focus)
        if csv_path is not None:
            focus_share = share_breakdown(load_range_kernel_totals(csv_path, NVTX_FORWARD))

    lines: list[str] = [
        "# Nsight Systems Profile Report",
        "",
        "Assignment 2 `nsys_profile` parts (a) and (b). Headless `nsys profile` + `nsys stats` only (no GUI).",
        "",
        "**Matrix:** model sizes `medium`, `xl`; context lengths `{256, 512, 1024}`; batch=4; vocab=10000; warmup=5; measure steps=10.",
        "",
        "**Method:** one profiled run per cell with nested NVTX — outer `forward_backward` (forward→loss→backward), inner `forward` (model + `cuda.synchronize`). Warmup uses different NVTX names and is ignored when reading `forward` / `forward_backward` rows. Kernel tables from `nvtx_kern_sum`; forward wall time from `nvtx_pushpop_sum` Avg of `forward`. Call counts reported per single forward (= Kern Inst / 10). Nsight Systems 2024.2.3 CLI (stock 2022.4 on this host cannot decode CUDA 12.4 kernels).",
        "",
        "## (a) Forward time: nsys vs Python",
        "",
        md_img(figure_a_rel, alt="part a", width=560),
        "",
        "| size | context | Python forward mean | nsys NVTX forward mean | ratio nsys/python |",
        "|---|---:|---:|---:|---:|",
    ]
    for size in NSYS_SIZES:
        for ctx in NSYS_CONTEXTS:
            cell = next((c for c in cells if c.model_size == size and c.context_length == ctx), None)
            if cell is None:
                lines.append(f"| {size} | {ctx} | — | — | — |")
            elif cell.oom:
                lines.append(f"| {size} | {ctx} | OOM | OOM | — |")
            else:
                py = cell.python_forward_mean_s
                ns = cell.nsys_forward_mean_s or float("nan")
                ratio = ns / py if py and math.isfinite(ns) else float("nan")
                lines.append(
                    f"| {size} | {ctx} | {_format_seconds(py)} | {_format_seconds(ns)} | {ratio:.3f} |"
                )

    if focus and focus.nsys_forward_mean_s is not None:
        py, ns = focus.python_forward_mean_s, focus.nsys_forward_mean_s
        lines += [
            "",
            f"**Answer (a):** On `medium` with context 512, nsys NVTX `forward` mean is {_format_seconds(ns)} versus Python `timeit`+`synchronize` mean {_format_seconds(py)} (ratio {ns / py:.3f}). They are the same order of magnitude; nsys is often slightly higher due to profiling overhead and range accounting.",
            "",
        ]
    else:
        lines += ["", "**Answer (a):** See table above for per-cell comparisons.", ""]

    lines += [
        "## (b) Top CUDA kernels",
        "",
        "### 前 5 名绝对耗时（原先那组图，保留）",
        "",
        "每个成功组合一张：横轴为该内核在 NVTX 范围内的**累计时间 (ms)**（10 次 measure step 合计）。"
        "总览多面板图如下；其下按组合拆成约 10 张单图（forward / step 各一）。",
        "",
        md_img(figure_b_rel, alt="part b top5 overview", width=720),
        "",
    ]
    for rel in top5_figure_rels:
        lines.append(md_img(rel, width=440))
        lines.append("")

    lines += [
        "### 占比饼图（第1 / 第2 / 第3–5合计 / 其余）",
        "",
        "百分比分母 = 该 NVTX 范围内**所有 CUDA 内核**的 Total Time 之和（不是只在前五名内部归一化）。"
        "总览条形图 + 约 10 张分组合饼图。",
        "",
        md_img(figure_b_share_overview_rel, alt="part b share overview", width=560),
        "",
    ]
    for rel in share_figure_rels:
        lines.append(md_img(rel, width=400))
        lines.append("")

    lines += [
        "| size | context | top kernel (forward) | calls / forward | top kernel (fwd+bwd) | same? |",
        "|---|---:|---|---:|---|---|",
    ]
    for size in NSYS_SIZES:
        for ctx in NSYS_CONTEXTS:
            cell = next((c for c in cells if c.model_size == size and c.context_length == ctx), None)
            if cell is None:
                lines.append(f"| {size} | {ctx} | — | — | — | — |")
            elif cell.oom:
                lines.append(f"| {size} | {ctx} | OOM | — | OOM | — |")
            else:
                same = "yes" if cell.same_top_kernel else "no"
                calls = cell.top_kernel_forward_calls_per_step
                calls_s = "n/a" if calls is None else f"{calls:.1f}"
                lines.append(
                    f"| {size} | {ctx} | `{cell.top_kernel_forward}` | {calls_s} | `{cell.top_kernel_step}` | {same} |"
                )

    if focus and focus.top_kernel_forward and focus.top5_forward and focus_share is not None:
        steps = 10
        lines += [
            "",
            "### 详解：以 `medium`、上下文长度 512 的向前传播为例",
            "",
            "先补一点读名字时会撞到的背景（不需要背，扫一眼即可）：",
            "",
            "- **Ampere（安培）**：NVIDIA 的一代 GPU 架构代号。你这台 A800 就属于 Ampere 一代。"
            "库函数名字里带 `ampere_`，表示这份矩阵乘实现是按 Ampere 硬件切出来的。",
            "- **kernel（内核）**：丢到 GPU 上跑的一小段程序。下面表里每一行就是一种这样的小程序。",
            "- **SGEMM**：Single-precision General Matrix Multiply，也就是 **FP32 的通用矩阵乘** "
            "（`C = A×B` 这类）。Transformer 里的线性层、注意力里的大矩阵乘，最后几乎都变成它。",
            "- **名字里的 `128x128` / `128x32`**：矩阵乘不会一口气算整张大矩阵，而是切成小块（tile）计算；"
            "这两个数字是切块大小，由 cuBLAS 等库按矩阵形状自动选。",
            "- **后缀 `tn` / `nn`**：描述相乘时两个输入要不要转置。"
            "`t` = transpose（转置），`n` = normal（不转置）。"
            "例如 `tn` = 左矩阵转置、右矩阵不转置。这只影响数据怎么摆，**本质仍是矩阵乘**。",
            "",
            f"以「向前」范围内**全部 CUDA 内核时间**为 100%："
            f"第1名占 **{focus_share['pct_rank1']:.1f}%**，"
            f"第2名占 **{focus_share['pct_rank2']:.1f}%**，"
            f"第3–5名合计占 **{focus_share['pct_rank345']:.1f}%**，"
            f"其余内核占 **{focus_share['pct_rest']:.1f}%**。"
            f"所有名字里带 gemm/sgemm 的矩阵乘合计约 **{focus_share['pct_gemm']:.1f}%**；"
            f"逐元素类小算子合计约 **{focus_share['pct_elementwise']:.1f}%**。",
            "",
            "下表仍列出前 5 名的绝对时间（10 次正式测量加总；除以 10 可粗看作单次向前分摊）。",
            "",
            "| 排名 | 累计耗时（10 次向前） | 约合每次向前 | 占全部 CUDA 时间 | 每次向前调用次数 | 内核（简称） |",
            "|---:|---:|---:|---:|---:|---|",
        ]
        for i, item in enumerate(focus.top5_forward, start=1):
            name = item["kernel"]
            # pct vs all kernels
            pct = next((r["pct"] for r in focus_share["ranked"] if r["kernel"] == name), float("nan"))
            if name.startswith("void at::native::"):
                if "MulFunctor" in name and "vectorized_elementwise" in name:
                    short = "逐元素乘法（向量化版）"
                elif "MulFunctor" in name and "elementwise_kernel" in name:
                    short = "逐元素乘法（普通版）"
                else:
                    short = f"`{name[:80]}…`"
            else:
                short = f"`{name}`"
            lines.append(
                f"| {i} | {_format_seconds(item['total_s'])} | "
                f"{_format_seconds(item['total_s'] / steps)} | "
                f"{pct:.1f}% | "
                f"{item['kern_inst'] / steps:.1f} | {short} |"
            )

        # list notable non-gemm kernels with pct
        non_gemm = [r for r in focus_share["ranked"] if not _is_gemm(r["kernel"])][:6]
        lines += [
            "",
            "**这五个分别在干什么：**",
            "",
            f"1. **`{focus.top5_forward[0]['kernel']}`**  "
            "Ampere 上的 FP32 大矩阵乘，切块 `128×128`，布局 `tn`。"
            "对应模型里一类很常见的稠密线性变换 / 投影（把一张激活矩阵乘上一块权重）。"
            f"**占全部 CUDA 时间的 {focus_share['pct_rank1']:.1f}%，是最大的一块。**",
            "",
            f"2. **`{focus.top5_forward[1]['kernel']}`**  "
            "还是 FP32 矩阵乘，只是切块改成了 `128×32`。库根据矩阵高宽选了另一种切法——"
            "多半对应另一类形状的线性层或注意力里的矩阵乘（例如宽度不同的投影）。"
            f"和第一名是「同一工种、不同刀法」，占 **{focus_share['pct_rank2']:.1f}%**。",
            "",
            f"3. **`{focus.top5_forward[2]['kernel']}`**  "
            "仍然是 FP32 矩阵乘，布局变成 `nn`（两边都不转置）。"
            "常见于注意力里「分数矩阵 × 值矩阵」这类两边都不转置的乘。**还是矩阵乘，不是别的算子。**",
            "",
            "4. **逐元素乘法（向量化版）**  "
            "PyTorch/ATen 生成的「每个元素各自乘一下」的小程序，并做了向量化。"
            "典型来源：注意力分数乘 `1/sqrt(d_k)`、和 mask 相关的逐元素乘等。"
            "**不是矩阵乘**，算量比 GEMM 小得多。",
            "",
            "5. **逐元素乘法（普通版）**  "
            "同样是逐元素乘，只是另一种启动配置。一次向前会 launch 很多次（这里约 290 次），"
            "但每次都很短，所以累计时间仍排在矩阵乘后面。",
            "",
            "**「小算子」具体是谁、占多少？**  "
            f"在 `medium@512` 向前范围内，非矩阵乘里最显眼的是各类 **逐元素乘法 / elementwise** "
            f"（合计约 **{focus_share['pct_elementwise']:.1f}%**）。其中较靠前的包括：",
            "",
        ]
        for r in non_gemm:
            lines.append(
                f"- `{_short_kernel_label(r['kernel'])}`：约 **{r['pct']:.1f}%** "
                f"（{_format_seconds(r['total_s'])} / 10 次向前合计）"
            )
        lines += [
            "",
            "它们调用可以很勤，但每个元素只做一次乘或加减，总浮点运算远小于大矩阵乘，"
            "所以单看百分比也压不过头部的 SGEMM。",
            "",
            "**谁占用时间最多？为什么？**  "
            f"第一名 `{focus.top5_forward[0]['kernel']}` 独占约 **{focus_share['pct_rank1']:.1f}%**；"
            f"前两名矩阵乘合起来约 **{focus_share['pct_rank1'] + focus_share['pct_rank2']:.1f}%**；"
            f"所有 GEMM 合计约 **{focus_share['pct_gemm']:.1f}%**。"
            "`medium` 有 24 层、隐藏维 1024、前馈维 4096，一层里多次线性投影和注意力矩阵乘，"
            "绝大部分算力都堆在这些 GEMM 上。",
            "",
            "**加上反向传播之后呢？**  "
            f"外层「向前+损失+向后」统计里，累计最久的内核"
            f"{'仍然是同一个名字' if focus.same_top_kernel else '换成了另一个名字'}："
            f"`{focus.top_kernel_step}`。",
            "",
            "### 两点启示（做完 A/B 之后）",
            "",
            "1. **秒表和 profiler 对得上，才说明你量的是同一件事。** "
            "Part (a) 里 Python 掐表和 nsys 的「向前」区间平均耗时只差大约 1% 量级——"
            "两边都在同步后看整段向前，结论可以互相印证。"
            "若差出一大截，优先怀疑：热机没丢掉、标签包错了范围、或把分析开销误当成模型时间。",
            "",
            "2. **优化应先盯矩阵乘：用占比说话，别只看调用次数。** "
            f"以 `medium@512` 向前为例：第1名 SGEMM 占全部 CUDA 时间的 "
            f"**{focus_share['pct_rank1']:.1f}%**，第2名再占 **{focus_share['pct_rank2']:.1f}%**，"
            f"第3–5名合计 **{focus_share['pct_rank345']:.1f}%**；"
            f"所有矩阵乘合计约 **{focus_share['pct_gemm']:.1f}%**。"
            f"相对地，逐元素乘法等小算子合计约 **{focus_share['pct_elementwise']:.1f}%**——"
            "哪怕 launch 上百次，也远小于头部 GEMM。"
            "所以后面做混合精度、FlashAttention、更快的 matmul，针对的都是这七八成时间的大头；"
            "别被「调用很勤」的小内核带偏优先级。",
            "",
        ]
    else:
        lines += ["", "**Answer (b):** 见上表。", ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite() -> None:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = ARTIFACTS_ROOT / f"suite_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    cells: list[CellResult] = []
    total = len(NSYS_SIZES) * len(NSYS_CONTEXTS)
    idx = 0
    for size in NSYS_SIZES:
        if size not in MODEL_SIZE_PRESETS:
            raise ValueError(size)
        for ctx in NSYS_CONTEXTS:
            idx += 1
            print(f"\n======== SUITE {idx}/{total}: {size} ctx={ctx} ========", flush=True)
            cell_dir = suite_dir / f"{size}_ctx{ctx}"
            try:
                cell = run_cell(size, ctx, cell_dir)
            except Exception as err:  # noqa: BLE001
                print(f"[suite] FAILED {size} ctx={ctx}: {err}", flush=True)
                cell = CellResult(
                    model_size=size,
                    context_length=ctx,
                    python_forward_mean_s=float("nan"),
                    python_forward_std_s=float("nan"),
                    nsys_forward_mean_s=None,
                    nsys_forward_instances=None,
                    top_kernel_forward=None,
                    top_kernel_forward_calls_per_step=None,
                    top_kernel_forward_total_s=None,
                    top_kernel_step=None,
                    top_kernel_step_total_s=None,
                    same_top_kernel=None,
                    top5_forward=[],
                    top5_step=[],
                    oom=False,
                    artifact_dir=_rel(cell_dir),
                    notes=f"failed: {err}",
                )
            cells.append(cell)
            print(f"[suite] done {size} ctx={ctx} oom={cell.oom} notes={cell.notes!r}", flush=True)

    plot_part_a(cells, FIGURE_A)
    top5_rels, share_rels = plot_part_b(cells, FIGURE_B, FIGURE_B_SHARE_OVERVIEW)
    write_report(
        cells,
        REPORT_PATH,
        top5_figure_rels=top5_rels,
        share_figure_rels=share_rels,
    )
    manifest = {
        "suite_dir": str(suite_dir.relative_to(REPO_ROOT)),
        "cells": [c.to_dict() for c in cells],
        "report": "reports/nsys-profile.md",
        "figure_a": "reports/figures/nsys_a_forward_python_vs_nsys.png",
        "figure_b": "reports/figures/nsys_b_top_kernels.png",
        "figure_b_share_overview": "reports/figures/nsys_b_share_overview.png",
        "figure_b_top5": top5_rels,
        "figure_b_shares": share_rels,
    }
    SUITE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[suite] report → reports/nsys-profile.md", flush=True)
    print("[suite] figures → reports/figures/nsys_a_*.png, nsys_b_*.png", flush=True)


def replot_from_manifest(manifest_path: Path | None = None) -> None:
    """Regenerate plots + report from an existing suite_manifest.json (no re-profile)."""
    path = manifest_path or SUITE_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cells = [CellResult.from_dict(c) for c in manifest["cells"]]
    plot_part_a(cells, FIGURE_A)
    top5_rels, share_rels = plot_part_b(cells, FIGURE_B, FIGURE_B_SHARE_OVERVIEW)
    write_report(
        cells,
        REPORT_PATH,
        top5_figure_rels=top5_rels,
        share_figure_rels=share_rels,
    )
    manifest["figure_b"] = "reports/figures/nsys_b_top_kernels.png"
    manifest["figure_b_share_overview"] = "reports/figures/nsys_b_share_overview.png"
    manifest["figure_b_top5"] = top5_rels
    manifest["figure_b_shares"] = share_rels
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    suite_rel = manifest.get("suite_dir")
    if suite_rel:
        suite_manifest = REPO_ROOT / suite_rel / "manifest.json"
        if suite_manifest.exists():
            suite_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[replot] report → reports/nsys-profile.md", flush=True)
    print(f"[replot] top5={len(top5_rels)} share={len(share_rels)}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="nsys_profile parts (a)+(b)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_work = sub.add_parser("workload", help="NVTX-annotated forward+backward workload (under nsys)")
    sp_work.add_argument("--model-size", required=True, choices=list(MODEL_SIZE_PRESETS))
    sp_work.add_argument("--context-length", type=int, required=True)
    sp_work.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    sp_work.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    sp_work.add_argument("--seed", type=int, default=DEFAULT_SEED)

    sp_py = sub.add_parser("python-baseline", help="Python forward timing without nsys")
    sp_py.add_argument("--model-size", required=True, choices=list(MODEL_SIZE_PRESETS))
    sp_py.add_argument("--context-length", type=int, required=True)
    sp_py.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    sp_py.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    sp_py.add_argument("--seed", type=int, default=DEFAULT_SEED)

    sub.add_parser("suite", help="Run full (a)+(b) matrix, plots, report")
    sub.add_parser("replot", help="Regenerate plots/report from suite_manifest.json (no GPU)")

    args = p.parse_args(argv)
    if args.cmd == "workload":
        cfg = _cfg(args.model_size, args.context_length, args.warmup, args.steps)
        cfg.seed = args.seed
        run_nvtx_workload(cfg)
    elif args.cmd == "python-baseline":
        cfg = _cfg(args.model_size, args.context_length, args.warmup, args.steps)
        cfg.seed = args.seed
        out = _run_python_forward_baseline_inplace(cfg)
        print(json.dumps(out, indent=2))
    elif args.cmd == "suite":
        run_suite()
    elif args.cmd == "replot":
        replot_from_manifest()
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
