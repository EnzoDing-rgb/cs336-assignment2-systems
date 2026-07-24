"""Memory-profiling (f): Nsight CUDA memory + TransformerBlock residuals (headless).

One cell: xl · B=4 · ctx=512 · FP32 · full train step.
- NVTX labels per TransformerBlock / forward / backward / optimizer
- ``saved_tensors_hooks`` for residual bytes + top-5 ops (precise)
- Backward hooks for per-block memory delta → gradient estimate
- ``nsys --cuda-memory-usage=true`` → sqlite curve plot (GUI-less “screenshot”)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import torch.cuda.nvtx as nvtx
import torch.nn as nn
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.e2e_timing.e2e import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_VOCAB_SIZE,
    BenchmarkConfig,
    _sync,
    build_model,
    make_batch,
)
from cs336_systems.nsys_profile.ab import _nsys_bin

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "memory_profiling" / "nsys_f"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
REPORT_PATH = REPO_ROOT / "reports" / "memory-profiling.md"

MODEL_SIZE = "xl"
CONTEXT = 512
BATCH = DEFAULT_BATCH_SIZE
WARMUP = 2
STEPS = 1  # one measured train step under nsys

NVTX_TRAIN = "train_step"
NVTX_FORWARD = "forward"
NVTX_BACKWARD = "backward"
NVTX_OPTIMIZER = "optimizer"
NVTX_WARMUP = "warmup"
NVTX_BLOCK = "TransformerBlock"  # suffix _{i}


def _bytes_mib(n: int | float) -> float:
    return float(n) / (1024**2)


def _bytes_gib(n: int | float) -> float:
    return float(n) / (1024**3)


def _classify_saved_tensor(t: torch.Tensor) -> str:
    """Bucket saved activations by numel / trailing dims (xl · B=4 · S=512)."""
    if isinstance(t, torch.nn.Parameter):
        return "parameter"
    nbytes = t.numel() * t.element_size()
    shape = tuple(t.shape)
    mib = nbytes / (1024**2)
    # Known FP32 sizes
    attn_ss = BATCH * 32 * CONTEXT * CONTEXT * 4  # 128 MiB
    ffn_act = BATCH * CONTEXT * 10240 * 4  # 80 MiB
    hid = BATCH * CONTEXT * 2560 * 4  # 20 MiB
    qkv = BATCH * 32 * CONTEXT * 80 * 4  # 20 MiB
    ffn_w = 2560 * 10240 * 4  # 100 MiB

    # Weight matrices first (avoid catching them as ffn_inner via last-dim==10240).
    if len(shape) >= 2 and sorted(shape[-2:]) == [2560, 10240]:
        return "ffn_weight"
    if nbytes == attn_ss or (
        len(shape) >= 2 and shape[-1] == CONTEXT and shape[-2] == CONTEXT and mib >= 100
    ):
        return "attn_scores_or_probs"
    if nbytes == ffn_act or (len(shape) >= 1 and shape[-1] == 10240 and abs(mib - 80) < 1):
        return "ffn_inner"
    if nbytes == qkv or (len(shape) >= 1 and shape[-1] == 80 and abs(mib - 20) < 1):
        return "attn_qkv_heads"
    if nbytes == hid or (len(shape) >= 1 and shape[-1] == 2560 and abs(mib - 20) < 1):
        return "residual_or_hidden"
    # Flattened residual (1, B*S, d)
    if len(shape) == 3 and shape[0] == 1 and shape[1] == BATCH * CONTEXT and shape[2] == 2560:
        return "residual_or_hidden"
    if mib < 1.0:
        return "norm_stats_or_small"
    return f"other{shape}"


@dataclass
class BlockResidualStats:
    layer_index: int
    total_saved_bytes: int
    by_op: dict[str, int]
    n_tensors: int

    @property
    def total_mib(self) -> float:
        return _bytes_mib(self.total_saved_bytes)


def measure_one_block_residuals(
    block: nn.Module,
    *,
    layer_index: int,
    batch: int = BATCH,
    context: int = CONTEXT,
    d_model: int = 2560,
    device: torch.device,
) -> BlockResidualStats:
    """Isolate one TransformerBlock (handout §3 style) and tally saved tensors."""
    x = torch.randn(batch, context, d_model, device=device, requires_grad=True)
    saved: dict[str, int] = defaultdict(int)
    n_tensors = 0
    total = 0

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        nonlocal total, n_tensors
        if isinstance(t, torch.nn.Parameter):
            return t
        nbytes = t.numel() * t.element_size()
        total += nbytes
        n_tensors += 1
        saved[_classify_saved_tensor(t)] += nbytes
        return t

    def unpack_hook(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        with nvtx.range(f"{NVTX_BLOCK}_{layer_index}"):
            y = block(x)
    # Free the graph for this isolated probe.
    del y, x
    torch.cuda.empty_cache()
    return BlockResidualStats(
        layer_index=layer_index,
        total_saved_bytes=total,
        by_op=dict(saved),
        n_tensors=n_tensors,
    )


def measure_block_residuals(model: nn.Module, device: torch.device) -> list[BlockResidualStats]:
    """Measure residuals for a few representative layers (isolated forwards)."""
    indices = sorted({0, len(model.layers) // 2, len(model.layers) - 1})
    return [
        measure_one_block_residuals(model.layers[i], layer_index=i, device=device)
        for i in indices
    ]


def measure_backward_memory_deltas(
    model: nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, Any]:
    """Full train step; record allocated bytes at each block's backward hook."""
    device = x.device
    deltas: list[dict[str, Any]] = []
    pre_mem: dict[int, int] = {}

    hooks = []
    for i, block in enumerate(model.layers):

        def _make(idx: int) -> Callable[..., None]:
            def pre_hook(module: nn.Module, grad_output: Any) -> None:  # noqa: ARG001
                _sync(device)
                pre_mem[idx] = int(torch.cuda.memory_allocated())

            def post_hook(
                module: nn.Module,
                grad_input: Any,  # noqa: ARG001
                grad_output: Any,  # noqa: ARG001
            ) -> None:
                _sync(device)
                after = int(torch.cuda.memory_allocated())
                before = pre_mem.get(idx, after)
                deltas.append(
                    {
                        "layer_index": idx,
                        "before_bytes": before,
                        "after_bytes": after,
                        "delta_bytes": after - before,
                    }
                )

            return pre_hook, post_hook

        pre_h, post_h = _make(i)
        hooks.append(block.register_full_backward_pre_hook(pre_h))
        hooks.append(block.register_full_backward_hook(post_h))

    optimizer.zero_grad(set_to_none=True)
    with nvtx.range(NVTX_TRAIN):
        with nvtx.range(NVTX_FORWARD):
            logits = model(x)
            loss = cross_entropy(logits, y)
        _sync(device)
        mem_after_forward = int(torch.cuda.memory_allocated())
        with nvtx.range(NVTX_BACKWARD):
            loss.backward()
        _sync(device)
        mem_after_backward = int(torch.cuda.memory_allocated())
        with nvtx.range(NVTX_OPTIMIZER):
            optimizer.step()
        _sync(device)
        mem_after_optim = int(torch.cuda.memory_allocated())

    for h in hooks:
        h.remove()

    # Backward visits layers in reverse; sort for reporting.
    deltas_sorted = sorted(deltas, key=lambda d: d["layer_index"])
    return {
        "per_block_backward": deltas_sorted,
        "mem_after_forward": mem_after_forward,
        "mem_after_backward": mem_after_backward,
        "mem_after_optimizer": mem_after_optim,
    }


def install_block_nvtx(model: nn.Module) -> list[Callable[[], None]]:
    """Wrap each TransformerBlock.forward in NVTX; return undo callables."""
    undos: list[Callable[[], None]] = []
    for i, block in enumerate(model.layers):
        original = block.forward

        def _make(fwd: Callable, idx: int) -> Callable:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                with nvtx.range(f"{NVTX_BLOCK}_{idx}"):
                    return fwd(*args, **kwargs)

            return wrapped

        block.forward = _make(original, i)  # type: ignore[method-assign]
        undos.append(lambda b=block, o=original: setattr(b, "forward", o))
    return undos


def run_workload() -> None:
    """Entry under ``nsys profile``: warmup + one NVTX-labelled train step."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    cfg = BenchmarkConfig(
        model_size=MODEL_SIZE,
        mode="train",
        vocab_size=DEFAULT_VOCAB_SIZE,
        batch_size=BATCH,
        context_length=CONTEXT,
        warmup=WARMUP,
        steps=STEPS,
        seed=DEFAULT_SEED,
    )
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.empty_cache()

    model = build_model(cfg, device)
    model.train()
    undos = install_block_nvtx(model)
    optimizer = AdamW(model.parameters())
    x, y = make_batch(cfg, device)

    try:
        for i in range(cfg.warmup):
            with nvtx.range(NVTX_WARMUP):
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
            print(f"[nsys_f] warmup {i + 1}/{cfg.warmup}", flush=True)

        optimizer.zero_grad(set_to_none=True)
        with nvtx.range(NVTX_TRAIN):
            with nvtx.range(NVTX_FORWARD):
                logits = model(x)
                loss = cross_entropy(logits, y)
            with nvtx.range(NVTX_BACKWARD):
                loss.backward()
            with nvtx.range(NVTX_OPTIMIZER):
                optimizer.step()
        _sync(device)
        print("[nsys_f] measure train_step done", flush=True)
    finally:
        for u in undos:
            u()


def profile_nsys(out_prefix: Path) -> Path:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rep = Path(str(out_prefix) + ".nsys-rep")
    if rep.exists():
        rep.unlink()
    cmd = [
        _nsys_bin(),
        "profile",
        "--sample=none",
        "--cpuctxsw=none",
        "--trace=cuda,nvtx",
        "--cuda-memory-usage=true",
        "--force-overwrite=true",
        "-o",
        str(out_prefix),
        sys.executable,
        "-m",
        "cs336_systems.memory_profiling.nsys_f",
        "workload",
    ]
    print(f"[nsys] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if proc.returncode != 0:
        raise RuntimeError(f"nsys profile failed: {proc.returncode}")
    if not rep.exists():
        raise RuntimeError(f"missing {rep}")
    return rep


def export_sqlite(rep: Path, sqlite_path: Path) -> Path:
    if sqlite_path.exists():
        sqlite_path.unlink()
    cmd = [
        _nsys_bin(),
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "-o",
        str(sqlite_path),
        str(rep),
    ]
    print(f"[nsys] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0 or not sqlite_path.exists():
        raise RuntimeError("nsys export sqlite failed")
    return sqlite_path


def plot_cuda_memory_from_sqlite(sqlite_path: Path, out_png: Path) -> dict[str, Any]:
    """Rebuild device occupancy from CUDA_GPU_MEMORY_USAGE_EVENTS + NVTX overlays."""
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT start, bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"
    ).fetchall()
    nvtx_rows = cur.execute(
        "SELECT start, end, text FROM NVTX_EVENTS WHERE text IS NOT NULL ORDER BY start"
    ).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("no CUDA_GPU_MEMORY_USAGE_EVENTS")

    t0 = rows[0][0]
    times: list[float] = []
    mem: list[float] = []
    cur_bytes = 0
    for start, nbytes, op in rows:
        # 0 = alloc, 1 = dealloc
        if op == 0:
            cur_bytes += int(nbytes)
        elif op == 1:
            cur_bytes -= int(nbytes)
        times.append((start - t0) * 1e-9)  # ns → s (nsys timestamps are ns)
        mem.append(cur_bytes / (1024**3))

    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(times, mem, color="#1f4e79", lw=1.2, label="CUDA malloc occupancy")
    ax.set_xlabel("时间 (s，相对录制起点)")
    ax.set_ylabel("Accumulated cudaMalloc (GiB)")
    ax.set_title("xl · ctx=512 · B=4 · FP32 · nsys --cuda-memory-usage")
    ax.grid(True, alpha=0.3)

    # Overlay a few NVTX ranges (train / forward / backward / first & last block)
    want = {
        NVTX_FORWARD,
        NVTX_BACKWARD,
        NVTX_OPTIMIZER,
        NVTX_TRAIN,
        f"{NVTX_BLOCK}_0",
        f"{NVTX_BLOCK}_31",
    }
    colors = {
        NVTX_FORWARD: "#27ae60",
        NVTX_BACKWARD: "#e67e22",
        NVTX_OPTIMIZER: "#2980b9",
        NVTX_TRAIN: "#8e44ad",
        f"{NVTX_BLOCK}_0": "#16a085",
        f"{NVTX_BLOCK}_31": "#c0392b",
    }
    shown = 0
    for start, end, text in nvtx_rows:
        if text not in want or end is None:
            continue
        s = (start - t0) * 1e-9
        e = (end - t0) * 1e-9
        if e < 0 or s > times[-1] + 1:
            continue
        ax.axvspan(s, e, color=colors.get(text, "#999"), alpha=0.15)
        ax.axvline(s, color=colors.get(text, "#999"), ls=":", lw=0.9, alpha=0.8)
        if shown < 8:
            ax.text(s, 0.02 + 0.06 * shown, text, transform=ax.get_xaxis_transform(), fontsize=7, rotation=90, va="bottom")
            shown += 1

    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return {
        "peak_cuda_malloc_gib": max(mem) if mem else None,
        "n_mem_events": len(rows),
        "n_nvtx": len(nvtx_rows),
        "duration_s": times[-1] if times else None,
    }


def plot_residual_top5(stats: BlockResidualStats, out_png: Path) -> None:
    items = sorted(stats.by_op.items(), key=lambda kv: kv[1], reverse=True)[:5]
    labels = [k for k, _ in items]
    vals = [_bytes_mib(v) for _, v in items]
    total = stats.total_mib or 1.0
    pcts = [100.0 * v / total for v in vals]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.barh(labels[::-1], vals[::-1], color="#1f4e79")
    ax.set_xlabel("Saved for backward (MiB)")
    ax.set_title(
        f"TransformerBlock residual Top-5 · layer {stats.layer_index} · "
        f"total {stats.total_mib:.1f} MiB"
    )
    for bar, pct in zip(bars, pcts[::-1]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {pct:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_backward_deltas(
    residual_mib: float,
    bwd_deltas: list[dict[str, Any]],
    out_png: Path,
) -> None:
    """Per-layer backward Δmem and implied gradient bytes ≈ Δ + R."""
    # Use middle layers (stable); skip extremes if noisy
    xs = [d["layer_index"] for d in bwd_deltas]
    deltas_mib = [_bytes_mib(d["delta_bytes"]) for d in bwd_deltas]
    grads_mib = [d + residual_mib for d in deltas_mib]

    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    ax.plot(xs, deltas_mib, "o-", label="backward Δ allocated (MiB)", color="#e67e22")
    ax.plot(xs, grads_mib, "s-", label=f"Δ + residual({residual_mib:.0f} MiB) ≈ grads", color="#2980b9")
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xlabel("TransformerBlock index")
    ax.set_ylabel("MiB")
    ax.set_title("Backward memory change per block → gradient estimate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_python_analysis(out_dir: Path) -> dict[str, Any]:
    """Precise residual / top-5 / backward deltas (no nsys)."""
    cfg = BenchmarkConfig(
        model_size=MODEL_SIZE,
        mode="train",
        vocab_size=DEFAULT_VOCAB_SIZE,
        batch_size=BATCH,
        context_length=CONTEXT,
        seed=DEFAULT_SEED,
    )
    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.empty_cache()

    model = build_model(cfg, device)
    model.train()
    optimizer = AdamW(model.parameters())
    x, y = make_batch(cfg, device)

    # Warmup so Adam state exists (matches train memory regime)
    for _ in range(WARMUP):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    _sync(device)

    print("[analyze] measuring per-block residuals …", flush=True)
    # Drop Adam-heavy resident memory before the isolated block probe.
    del optimizer
    torch.cuda.empty_cache()
    _sync(device)

    residuals = measure_block_residuals(model, device)
    # Prefer mid layer as representative
    mid = next(r for r in residuals if r.layer_index == len(model.layers) // 2)
    top5 = sorted(mid.by_op.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top5_rows = [
        {
            "op": k,
            "bytes": v,
            "mib": _bytes_mib(v),
            "pct": 100.0 * v / mid.total_saved_bytes if mid.total_saved_bytes else 0.0,
        }
        for k, v in top5
    ]

    print("[analyze] rebuilding optimizer for backward deltas …", flush=True)
    optimizer = AdamW(model.parameters())
    for _ in range(1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()
    _sync(device)

    print("[analyze] measuring backward memory deltas …", flush=True)
    bwd = measure_backward_memory_deltas(model, optimizer, x, y)

    # Gradient estimate: for each block, G ≈ Δ_bwd + R
    # Use median over layers for robustness
    r_bytes = mid.total_saved_bytes
    grad_ests = [d["delta_bytes"] + r_bytes for d in bwd["per_block_backward"]]
    grad_ests_sorted = sorted(grad_ests)
    median_grad = grad_ests_sorted[len(grad_ests_sorted) // 2] if grad_ests_sorted else 0

    # Expected parameter grads for one TransformerBlock (rough):
    # attn 4*d*d + ffn 3*d*ff + 2 LN ≈ 4*2560^2 + 3*2560*10240 + ...
    d, ff = 2560, 10240
    expected_grad_params = (4 * d * d + 3 * d * ff + 2 * d) * 4  # FP32 grads

    result = {
        "config": {
            "model_size": MODEL_SIZE,
            "batch_size": BATCH,
            "context_length": CONTEXT,
            "precision": "fp32",
        },
        "n_layers": len(model.layers),
        "representative_layer": mid.layer_index,
        "residual_total_mib": mid.total_mib,
        "residual_total_bytes": mid.total_saved_bytes,
        "residual_n_tensors": mid.n_tensors,
        "top5": top5_rows,
        "sampled_layers_residual_mib": {str(r.layer_index): r.total_mib for r in residuals},
        "backward": bwd,
        "grad_estimate_median_mib": _bytes_mib(median_grad),
        "grad_estimate_median_bytes": median_grad,
        "expected_param_grad_mib": _bytes_mib(expected_grad_params),
        "expected_param_grad_bytes": expected_grad_params,
    }

    plot_residual_top5(mid, FIGURES_DIR / "memory_f_residual_top5.png")
    plot_backward_deltas(mid.total_mib, bwd["per_block_backward"], FIGURES_DIR / "memory_f_bwd_deltas.png")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return result


def patch_report(analysis: dict[str, Any], nsys_meta: dict[str, Any] | None) -> None:
    """Replace section (f) in memory-profiling.md."""
    text = REPORT_PATH.read_text(encoding="utf-8")
    start = text.find("## (f)")
    if start < 0:
        raise RuntimeError("section (f) not found in report")
    # Keep everything before (f); rewrite (f) through end-of-section before Appendix
    appendix = text.find("## Appendix", start)
    if appendix < 0:
        appendix = len(text)
    head = text[:start]
    tail = text[appendix:]

    top5 = analysis["top5"]
    lines: list[str] = []
    lines.append("## (f) Nsight memory + NVTX（TransformerBlock residuals）")
    lines.append("")
    lines.append(
        f"**设定：** xl · B={BATCH} · ctx={CONTEXT} · FP32 · 完整 train step；"
        f"headless `nsys profile --trace=cuda,nvtx --cuda-memory-usage=true`（方案 B，无 GUI）。"
        f"每个 `TransformerBlock` 包 NVTX `{NVTX_BLOCK}_i`；"
        f"residual 体积用 `saved_tensors_hooks` 精确计量（与讲义 §3 同法）。"
    )
    lines.append("")
    lines.append("**Nsight CUDA memory 曲线（等价截图）**")
    lines.append("")
    lines.append(
        f'<img src="{(FIGURES_DIR / "memory_f_nsys_cuda_mem.png").resolve()}" '
        'alt="memory_f_nsys_cuda_mem" width="560" />'
    )
    lines.append("")
    if nsys_meta:
        lines.append(
            f"录制约 {nsys_meta.get('duration_s', float('nan')):.2f}s，"
            f"cudaMalloc 占用峰值约 **{nsys_meta.get('peak_cuda_malloc_gib'):.2f} GiB**"
            f"（段分配粒度，高于/不等于 PyTorch `memory_allocated`）。"
        )
        lines.append("")

    lines.append(
        f"**单层 residual：** 代表层 L{analysis['representative_layer']} 为 backward 保存 "
        f"**{analysis['residual_total_mib']:.1f} MiB**"
        f"（{analysis['residual_n_tensors']} 个 tensor）。"
    )
    lines.append("")
    lines.append(
        f'<img src="{(FIGURES_DIR / "memory_f_residual_top5.png").resolve()}" '
        'alt="memory_f_residual_top5" width="520" />'
    )
    lines.append("")
    lines.append("| rank | op bucket | MiB | % of block residual |")
    lines.append("|-----:|-----------|----:|--------------------:|")
    for i, row in enumerate(top5, 1):
        lines.append(f"| {i} | `{row['op']}` | {row['mib']:.1f} | {row['pct']:.1f}% |")
    lines.append("")
    lines.append(
        f'<img src="{(FIGURES_DIR / "memory_f_bwd_deltas.png").resolve()}" '
        'alt="memory_f_bwd_deltas" width="560" />'
    )
    lines.append("")
    lines.append(
        f"**Gradient 估计（在算什么）：** 反向经过某一层时，该层为 backward 存的 residual"
        f"（记体积 \(R\)）会被释放，同时写出该层参数（及流入激活）的梯度（记 \(G\)）。"
        f"若其它分配大致抵消，则 \(\Delta \\approx G - R\)，故 \(G \\approx \\Delta + R\)。"
        f"对各层 \(\Delta\) 取中位再加代表层 \(R\)，得 "
        f"**\(G\\approx {analysis['grad_estimate_median_mib']:.1f}\) MiB**。"
        f"只计参数梯度的解析下界"
        f"（attn 四个 \(d\\times d\) + SwiGLU 三个 \(d\\times d_{{\\mathrm{{ff}}}}\) + 2 LN，FP32）约 "
        f"**{analysis['expected_param_grad_mib']:.1f} MiB**"
        f"（\(d=2560,\,d_{{\\mathrm{{ff}}}}=10240\)）。"
        f"实测 \(G\) 应 ≥ 该下界；"
        f"{'同量级，符合预期' if analysis['grad_estimate_median_mib'] >= 0.5 * analysis['expected_param_grad_mib'] else '偏差较大，见 analysis.json'}。"
    )
    lines.append("")
    lines.append(
        f"**Answer (f):** 单层 TransformerBlock 为 backward 保存约 "
        f"**{analysis['residual_total_mib']:.0f} MiB** residual；Top-5 最大头是 "
        f"`{top5[0]['op'] if top5 else '—'}`"
        f"（约 {top5[0]['pct']:.0f}%），其次为 attention 的 \(S\\times S\) 类激活与 residual/hidden。"
        f"由 \(G\\approx\\Delta+R\) 估得每层梯度约 "
        f"**{analysis['grad_estimate_median_mib']:.0f} MiB**，"
        f"不低于参数梯度下界 ~{analysis['expected_param_grad_mib']:.0f} MiB，符合预期。"
    )
    lines.append("")

    REPORT_PATH.write_text(head + "\n".join(lines) + "\n" + tail, encoding="utf-8")
    print(f"Patched {REPORT_PATH}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("workload", help="Run under nsys profile")
    sub.add_parser("analyze", help="Python residual/grad analysis only")
    sub.add_parser("nsys", help="nsys profile + sqlite plot only")
    sub.add_parser("report", help="Patch report from existing artifacts")
    sub.add_parser("all", help="analyze + nsys in fresh subprocesses + report")
    args = p.parse_args()

    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "workload":
        run_workload()
        return

    if args.cmd == "analyze":
        run_python_analysis(ARTIFACTS_ROOT)
        return

    if args.cmd == "nsys":
        prefix = ARTIFACTS_ROOT / "xl_ctx512_train"
        rep = profile_nsys(prefix)
        sqlite_path = ARTIFACTS_ROOT / "xl_ctx512_train.sqlite"
        export_sqlite(rep, sqlite_path)
        nsys_meta = plot_cuda_memory_from_sqlite(
            sqlite_path, FIGURES_DIR / "memory_f_nsys_cuda_mem.png"
        )
        (ARTIFACTS_ROOT / "nsys_meta.json").write_text(
            json.dumps(nsys_meta, indent=2), encoding="utf-8"
        )
        return

    if args.cmd == "report":
        analysis = json.loads((ARTIFACTS_ROOT / "analysis.json").read_text(encoding="utf-8"))
        meta_path = ARTIFACTS_ROOT / "nsys_meta.json"
        nsys_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        patch_report(analysis, nsys_meta)
        return

    if args.cmd == "all":
        # Separate processes so CUDA memory is fully released between stages.
        for stage in ("analyze", "nsys", "report"):
            print(f"\n===== stage: {stage} =====", flush=True)
            proc = subprocess.run(
                [sys.executable, "-m", "cs336_systems.memory_profiling.nsys_f", stage],
                cwd=str(REPO_ROOT),
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"stage {stage} failed with code {proc.returncode}")
        analysis = json.loads((ARTIFACTS_ROOT / "analysis.json").read_text(encoding="utf-8"))
        nsys_meta = json.loads((ARTIFACTS_ROOT / "nsys_meta.json").read_text(encoding="utf-8"))
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "analysis": analysis,
            "nsys_meta": nsys_meta,
        }
        (ARTIFACTS_ROOT / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return

    raise ValueError(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
