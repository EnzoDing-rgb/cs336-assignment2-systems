"""三对照组 FlashAttention 延迟基准 + tile 消融。

实验 A（主对比）
  naive attention / flash_pytorch / flash_triton
  两版 Flash 使用相同 heuristic tile（_choose_flash_tiles）
  同轮测量 forward / backward / e2e
  网格：S × d × {fp32, bf16}，B=1，causal=True

实验 B（tile 消融）
  仅 flash_pytorch vs flash_triton
  固定若干 (S, d)、fp32
  tile ∈ {(16,16),(32,32),(64,64),(128,128),heuristic}
  同轮 fwd / bwd / e2e —— 看加大 tile 对 PyTorch 能补多少、Triton 大 tile 是否挂
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton.testing

from cs336_systems.flash_attention.flash_attn_pytorch import (
    FlashAttention2PyTorchFunc,
    attention_reference,
)
from cs336_systems.flash_attention.flash_attn_triton import (
    FlashAttention2TritonFunc,
    _choose_flash_tiles,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "flash_attention_benchmark"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
REPORT_PATH = REPO_ROOT / "reports" / "flash-attention-benchmark.md"

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODEL = [16, 32, 64, 128]
DTYPES = [torch.float32, torch.bfloat16]
BATCH_SIZE = 1
IS_CAUSAL = True
SEED = 0

# 图例 / JSON 里的实现名
IMPL_NAIVE = "naive_attention"
IMPL_PT = "flash_pytorch"
IMPL_TRITON = "flash_triton"
MAIN_IMPLS = (IMPL_NAIVE, IMPL_PT, IMPL_TRITON)

# flash_pytorch 是 Python 双层 for + 多次小 kernel：tile 太小或 S 太大时，
# 单次前向可到秒～分钟级，do_bench 会把整轮实验拖死。下面两处是测得完的折中：
# - 主对比：flash_pytorch 只跑到 S<=4096（naive/triton 仍跑满网格）
# - tile 消融：大 S 只测较大 tile；小 tile 只在中等 S 上测
FLASH_PYTORCH_MAX_S_MAIN = 2048

TILE_ABLATION_POINTS = [
    # (seq_len, d_model)
    (2048, 64),
    (4096, 64),
    (8192, 64),
    (8192, 128),
]
# 每个 (S,d) 实际跑哪些 tile：大 S 去掉过小 tile，避免 PyTorch 循环爆炸
TILE_CHOICES_BY_POINT: dict[tuple[int, int], list[tuple[int, int] | None]] = {
    (2048, 64): [(16, 16), (32, 32), (64, 64), (128, 128), None],
    (4096, 64): [(32, 32), (64, 64), (128, 128), None],
    (8192, 64): [(64, 64), (128, 128), None],
    (8192, 128): [(64, 64), (128, 128), None],
}


@dataclass
class CellResult:
    experiment: str  # "main" | "tile"
    impl: str
    seq_len: int
    d_model: int
    dtype: str
    B_q: int | None
    B_k: int | None
    tile_label: str
    forward_ms: float | None
    backward_ms: float | None
    e2e_ms: float | None
    oom: bool
    error: str | None
    skipped: bool


def _dtype_name(dtype: torch.dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp32"


def _cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _make_inputs(seq_len: int, d_model: int, dtype: torch.dtype, requires_grad: bool):
    q = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=requires_grad)
    k = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=requires_grad)
    v = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", dtype=dtype, requires_grad=requires_grad)
    do = torch.randn(BATCH_SIZE, seq_len, d_model, device="cuda", dtype=dtype)
    return q, k, v, do


def _resolve_tiles(seq_len: int, d_model: int, tile: tuple[int, int] | None) -> tuple[int, int, str]:
    if tile is None:
        bq, bk = _choose_flash_tiles(seq_len, d_model)
        return bq, bk, f"heuristic({bq}x{bk})"
    bq, bk = tile
    return bq, bk, f"{bq}x{bk}"


def _forward(impl: str, q, k, v, B_q: int | None, B_k: int | None):
    if impl == IMPL_NAIVE:
        return attention_reference(q, k, v, is_causal=IS_CAUSAL)[0]
    if impl == IMPL_PT:
        return FlashAttention2PyTorchFunc.apply(q, k, v, IS_CAUSAL, B_q or 0, B_k or 0)
    if impl == IMPL_TRITON:
        return FlashAttention2TritonFunc.apply(q, k, v, IS_CAUSAL, B_q or 0, B_k or 0)
    raise ValueError(impl)


def _bench_ms(fn, *, heavy: bool = False) -> float:
    # flash_pytorch 单次可达秒～分钟级：和 Triton 比的是数量级，不需要高 rep。
    # warmup=1, rep=3 足够；再高只是线性浪费墙钟时间。
    if heavy:
        return float(triton.testing.do_bench(fn, warmup=1, rep=3))
    return float(triton.testing.do_bench(fn, warmup=5, rep=20))


def _run_one(
    experiment: str,
    impl: str,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    tile: tuple[int, int] | None,
) -> CellResult:
    if impl == IMPL_NAIVE:
        B_q = B_k = None
        tile_label = "n/a"
    else:
        B_q, B_k, tile_label = _resolve_tiles(seq_len, d_model, tile)

    base = dict(
        experiment=experiment,
        impl=impl,
        seq_len=seq_len,
        d_model=d_model,
        dtype=_dtype_name(dtype),
        B_q=B_q,
        B_k=B_k,
        tile_label=tile_label,
        forward_ms=None,
        backward_ms=None,
        e2e_ms=None,
        oom=False,
        error=None,
        skipped=False,
    )
    try:
        _cleanup()
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        q, k, v, _ = _make_inputs(seq_len, d_model, dtype, requires_grad=False)

        heavy = impl == IMPL_PT

        def fwd():
            return _forward(impl, q, k, v, B_q, B_k)

        fwd()
        torch.cuda.synchronize()
        base["forward_ms"] = _bench_ms(fwd, heavy=heavy)
        del q, k, v
        _cleanup()

        q, k, v, do = _make_inputs(seq_len, d_model, dtype, requires_grad=True)
        out = _forward(impl, q, k, v, B_q, B_k)
        torch.cuda.synchronize()

        def bwd():
            out.backward(do, retain_graph=True)
            q.grad = None
            k.grad = None
            v.grad = None

        bwd()
        torch.cuda.synchronize()
        base["backward_ms"] = _bench_ms(bwd, heavy=heavy)
        del q, k, v, do, out
        _cleanup()

        q, k, v, do = _make_inputs(seq_len, d_model, dtype, requires_grad=True)

        def e2e():
            o = _forward(impl, q, k, v, B_q, B_k)
            o.backward(do)
            q.grad = None
            k.grad = None
            v.grad = None

        e2e()
        torch.cuda.synchronize()
        base["e2e_ms"] = _bench_ms(e2e, heavy=heavy)
        del q, k, v, do
        _cleanup()
        return CellResult(**base)
    except torch.cuda.OutOfMemoryError as e:
        _cleanup()
        base["oom"] = True
        base["error"] = f"OOM: {e}"
        return CellResult(**base)
    except Exception as e:  # noqa: BLE001 — benchmark must record failures
        _cleanup()
        base["error"] = f"{type(e).__name__}: {e}"
        return CellResult(**base)


def run_main_sweep(smoke: bool = False) -> list[CellResult]:
    seqs = [128, 512, 2048] if smoke else SEQ_LENS
    dims = [64] if smoke else D_MODEL
    dtypes = [torch.float32] if smoke else DTYPES
    rows: list[CellResult] = []
    # 某 (impl,d,dtype) 一旦 OOM，更大 S 直接 skip
    dead: set[tuple[str, int, str]] = set()

    total = len(MAIN_IMPLS) * len(seqs) * len(dims) * len(dtypes)
    done = 0
    t0 = time.time()
    for dtype in dtypes:
        for d_model in dims:
            for seq_len in seqs:
                for impl in MAIN_IMPLS:
                    done += 1
                    key = (impl, d_model, _dtype_name(dtype))
                    if impl == IMPL_PT and seq_len > FLASH_PYTORCH_MAX_S_MAIN:
                        rows.append(
                            CellResult(
                                experiment="main",
                                impl=impl,
                                seq_len=seq_len,
                                d_model=d_model,
                                dtype=_dtype_name(dtype),
                                B_q=None,
                                B_k=None,
                                tile_label="skipped_slow",
                                forward_ms=None,
                                backward_ms=None,
                                e2e_ms=None,
                                oom=False,
                                error=None,
                                skipped=True,
                            )
                        )
                        print(
                            f"[{done}/{total}] SKIP {impl} S={seq_len} d={d_model} {_dtype_name(dtype)} "
                            f"(python-tile too slow for S>{FLASH_PYTORCH_MAX_S_MAIN})",
                            flush=True,
                        )
                        continue
                    if key in dead:
                        rows.append(
                            CellResult(
                                experiment="main",
                                impl=impl,
                                seq_len=seq_len,
                                d_model=d_model,
                                dtype=_dtype_name(dtype),
                                B_q=None,
                                B_k=None,
                                tile_label="n/a" if impl == IMPL_NAIVE else "skipped",
                                forward_ms=None,
                                backward_ms=None,
                                e2e_ms=None,
                                oom=False,
                                error=None,
                                skipped=True,
                            )
                        )
                        print(f"[{done}/{total}] SKIP {impl} S={seq_len} d={d_model} {_dtype_name(dtype)}", flush=True)
                        continue
                    print(
                        f"[{done}/{total}] main {impl} S={seq_len} d={d_model} {_dtype_name(dtype)} ...",
                        flush=True,
                    )
                    cell = _run_one("main", impl, seq_len, d_model, dtype, tile=None)
                    rows.append(cell)
                    status = "OOM" if cell.oom else ("ERR" if cell.error else "ok")
                    print(
                        f"  -> {status} tile={cell.tile_label} "
                        f"fwd={cell.forward_ms} bwd={cell.backward_ms} e2e={cell.e2e_ms}",
                        flush=True,
                    )
                    if cell.oom:
                        dead.add(key)
    print(f"main sweep done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


def run_tile_ablation(smoke: bool = False) -> list[CellResult]:
    if smoke:
        points = [(2048, 64)]
        tile_map = {(2048, 64): [(16, 16), (64, 64), None]}
    else:
        points = TILE_ABLATION_POINTS
        tile_map = TILE_CHOICES_BY_POINT
    impls = (IMPL_PT, IMPL_TRITON)
    dtype = torch.float32
    rows: list[CellResult] = []
    total = sum(len(tile_map[p]) * len(impls) for p in points)
    done = 0
    t0 = time.time()
    for seq_len, d_model in points:
        for tile in tile_map[(seq_len, d_model)]:
            for impl in impls:
                done += 1
                label = "heuristic" if tile is None else f"{tile[0]}x{tile[1]}"
                print(
                    f"[{done}/{total}] tile {impl} S={seq_len} d={d_model} tile={label} ...",
                    flush=True,
                )
                cell = _run_one("tile", impl, seq_len, d_model, dtype, tile=tile)
                rows.append(cell)
                status = "OOM" if cell.oom else ("ERR" if cell.error else "ok")
                print(
                    f"  -> {status} resolved={cell.tile_label} "
                    f"fwd={cell.forward_ms} bwd={cell.backward_ms} e2e={cell.e2e_ms}",
                    flush=True,
                )
    print(f"tile ablation done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_IMPL_STYLE = {
    IMPL_NAIVE: dict(linestyle="-", marker="o", linewidth=1.9, label="naive attention"),
    IMPL_PT: dict(linestyle="--", marker="s", linewidth=1.6, label="flash_pytorch"),
    IMPL_TRITON: dict(linestyle=":", marker="D", linewidth=1.8, label="flash_triton"),
}


def _main_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("experiment", "main") == "main"]


def _tile_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("experiment") == "tile"]


def make_figures(rows: list[dict]) -> list[Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "--",
        }
    )
    paths: list[Path] = []
    main = _main_rows(rows)
    d_colors = {16: "#4C78A8", 32: "#F58518", 64: "#54A24B", 128: "#E45756"}

    def series_vs_s(impl: str, dtype: str, metric: str, d_model: int):
        xs, ys = [], []
        for r in main:
            if (
                r["impl"] == impl
                and r["dtype"] == dtype
                and r["d_model"] == d_model
                and not r.get("oom")
                and not r.get("skipped")
                and r.get(metric) is not None
            ):
                xs.append(r["seq_len"])
                ys.append(r[metric])
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        return [xs[i] for i in order], [ys[i] for i in order]

    # --- main: vs S ---
    for metric, ylab, fname, title in (
        ("forward_ms", "Forward latency (ms)", "flash_bench_forward_vs_seq.png", "Forward vs sequence length"),
        ("backward_ms", "Backward latency (ms)", "flash_bench_backward_vs_seq.png", "Backward vs sequence length"),
        ("e2e_ms", "End-to-end latency (ms)", "flash_bench_e2e_vs_seq.png", "End-to-end vs sequence length"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
        for ax, dtype in zip(axes, ("fp32", "bf16")):
            for d in D_MODEL:
                color = d_colors[d]
                for impl in MAIN_IMPLS:
                    xs, ys = series_vs_s(impl, dtype, metric, d)
                    if not xs:
                        continue
                    st = dict(_IMPL_STYLE[impl])
                    lab = st.pop("label")
                    ax.plot(xs, ys, color=color, markersize=4.5, label=f"{lab} d={d}", **st)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("sequence length $S$")
            ax.set_ylabel(ylab)
            ax.set_title(f"{title} · {dtype}")
            ax.legend(ncols=2, frameon=True, fancybox=False, edgecolor="#cccccc", fontsize=7)
        fig.suptitle(
            "Solid+circle = naive attention · Dashed+square = flash_pytorch · Dotted+diamond = flash_triton · Color = $d$",
            fontsize=9,
            y=1.02,
        )
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    # --- main: vs d (selected S) ---
    seq_for_d = [512, 2048, 8192]
    s_colors = {512: "#4C78A8", 2048: "#F58518", 8192: "#54A24B"}

    def point(impl, dtype, metric, d_model, seq_len):
        for r in main:
            if (
                r["impl"] == impl
                and r["dtype"] == dtype
                and r["d_model"] == d_model
                and r["seq_len"] == seq_len
                and not r.get("oom")
                and not r.get("skipped")
                and r.get(metric) is not None
            ):
                return r[metric]
        return None

    for metric, ylab, fname, title in (
        ("forward_ms", "Forward latency (ms)", "flash_bench_forward_vs_d.png", "Forward vs embedding dim"),
        ("backward_ms", "Backward latency (ms)", "flash_bench_backward_vs_d.png", "Backward vs embedding dim"),
        ("e2e_ms", "End-to-end latency (ms)", "flash_bench_e2e_vs_d.png", "End-to-end vs embedding dim"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
        for ax, dtype in zip(axes, ("fp32", "bf16")):
            for s in seq_for_d:
                color = s_colors[s]
                for impl in MAIN_IMPLS:
                    xs, ys = [], []
                    for d in D_MODEL:
                        y = point(impl, dtype, metric, d, s)
                        if y is not None:
                            xs.append(d)
                            ys.append(y)
                    if not xs:
                        continue
                    st = dict(_IMPL_STYLE[impl])
                    lab = st.pop("label")
                    ax.plot(xs, ys, color=color, markersize=5, label=f"{lab} S={s}", **st)
            ax.set_xticks(D_MODEL)
            ax.set_xlabel("embedding dim $d$")
            ax.set_ylabel(ylab)
            ax.set_yscale("log")
            ax.set_title(f"{title} · {dtype}")
            ax.legend(ncols=2, frameon=True, fancybox=False, edgecolor="#cccccc", fontsize=7)
        fig.suptitle(
            "Solid = naive attention · Dashed = flash_pytorch · Dotted = flash_triton · Color = $S$",
            fontsize=9,
            y=1.02,
        )
        out = FIGURES_DIR / fname
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    # --- tile ablation: latency vs tile size ---
    tile_data = _tile_rows(rows)
    if tile_data:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
        metrics = (
            ("forward_ms", "Forward (ms)"),
            ("backward_ms", "Backward (ms)"),
            ("e2e_ms", "End-to-end (ms)"),
        )
        # x-axis order by numeric tile area; heuristic last per point plotted separately
        point_colors = {
            (2048, 64): "#4C78A8",
            (4096, 64): "#F58518",
            (8192, 64): "#54A24B",
            (8192, 128): "#E45756",
        }
        for ax, (metric, ylab) in zip(axes, metrics):
            for impl, ls, mk in ((IMPL_PT, "--", "s"), (IMPL_TRITON, ":", "D")):
                for (S, d), color in point_colors.items():
                    pts = []
                    for r in tile_data:
                        if (
                            r["impl"] == impl
                            and r["seq_len"] == S
                            and r["d_model"] == d
                            and not r.get("oom")
                            and not r.get("skipped")
                            and r.get(metric) is not None
                            and r.get("B_q") is not None
                        ):
                            # skip pure heuristic label duplication: still plot by resolved B_q
                            pts.append((r["B_q"], r[metric], r["tile_label"]))
                    # dedupe by B_q (heuristic may collide with a fixed size)
                    by_bq: dict[int, float] = {}
                    for bq, y, _lab in pts:
                        by_bq[bq] = y
                    if not by_bq:
                        continue
                    xs = sorted(by_bq)
                    ys = [by_bq[x] for x in xs]
                    ax.plot(
                        xs,
                        ys,
                        linestyle=ls,
                        marker=mk,
                        color=color,
                        markersize=5,
                        linewidth=1.6,
                        label=f"{'pt' if impl == IMPL_PT else 'triton'} S={S} d={d}",
                    )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xlabel("tile $B_q$ (= $B_k$)")
            ax.set_ylabel(ylab)
            ax.set_title(ylab)
            ax.legend(fontsize=6, ncols=1, frameon=True, fancybox=False, edgecolor="#cccccc")
        fig.suptitle(
            "Tile ablation (fp32) · Dashed = flash_pytorch · Dotted = flash_triton · Color = (S, d)",
            fontsize=9,
            y=1.03,
        )
        out = FIGURES_DIR / "flash_bench_tile_ablation.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(out)

    return paths


def _fmt_ms(x: float | None, oom: bool, skipped: bool, err: str | None = None) -> str:
    if x is not None:
        if x >= 100:
            return f"{x:.0f}"
        if x >= 10:
            return f"{x:.1f}"
        return f"{x:.2f}"
    if skipped:
        return "skip"
    if oom:
        return "OOM"
    if err:
        return "err"
    return "—"


def write_report(rows: list[dict], figure_paths: list[Path]) -> None:
    """中文报告：设置 → 每图解析 → 文末总结论（结论不前置）。"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    main = _main_rows(rows)
    tile = _tile_rows(rows)

    by: dict[tuple, dict] = {}
    for r in main:
        by[(r["impl"], r["seq_len"], r["d_model"], r["dtype"])] = r

    def ms(impl: str, S: int, d: int, dtype: str, metric: str) -> float | None:
        r = by.get((impl, S, d, dtype))
        if r is None or r.get("skipped"):
            return None
        return r.get(metric)

    lines: list[str] = []
    lines.append("# FlashAttention 三对照组基准测试报告")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append(f"- GPU：`{gpu}`")
    lines.append("- Batch = 1，**causal = True**")
    lines.append("- 计时：`triton.testing.do_bench` 中位数（ms）；每格同轮测 forward / backward / e2e")
    lines.append("")
    lines.append("### 1.1 三对照组（不是三个 baseline）")
    lines.append("")
    lines.append("| 名称 | 前向 | 反向 |")
    lines.append("|------|------|------|")
    lines.append("| **naive attention** | 物化 $S\\times S$ 的朴素实现 | 普通 autograd |")
    lines.append(
        "| **flash_pytorch** | PyTorch 分块 Flash（Algorithm 1） | `torch.compile` dense 重算 $P$（Eq.13–19） |"
    )
    lines.append("| **flash_triton** | Triton 融合分块前向 | Triton 分块反向（Algorithm 2） |")
    lines.append("")
    lines.append(
        "主对比里两版 Flash 使用 **相同** `_choose_flash_tiles(S,d)`，"
        "因此主图上的差距应归因于执行引擎（融合/片上 vs Python+多 kernel / dense 反向），而不是分块大小不同。"
    )
    lines.append("")
    lines.append("### 1.2 两轮实验")
    lines.append("")
    lines.append(
        "- **实验 A（主对比）**：$S\\in\\{128\\ldots65536\\}$，$d\\in\\{16,32,64,128\\}$，dtype $\\in$ `{fp32,bf16}`；"
        "某 `(impl,d,dtype)` OOM 后更大 $S$ skip。"
        f"例外：`flash_pytorch` 主对比只测到 $S\\le {FLASH_PYTORCH_MAX_S_MAIN}$——"
        "Python 双层 tile 循环在更长序列上单次前向可达秒级，do_bench 无法在合理时间内完成；"
        "更长 $S$ 上的 PyTorch 行为改由实验 B（更大 tile）观察。"
    )
    lines.append(
        "- **实验 B（tile 消融）**：仅两版 Flash；fp32；"
        "$(S,d)\\in\\{(2048,64),(4096,64),(8192,64),(8192,128)\\}$；"
        "中等 $S$ 扫较全的 tile 集合，较大 $S$ 去掉过小 tile（否则 PyTorch 双层循环不可测）。"
        "用来回答：PyTorch 加大 tile 能补多少；Triton 大 tile 是否因 shared memory 失败。"
    )
    lines.append("")
    lines.append(
        "**读图**：实线+圆 = naive attention；虚线+方 = flash_pytorch；点线+菱 = flash_triton。"
        "对 $S$ 图颜色 = $d$；对 $d$ 图颜色 = $S$。左 fp32，右 bf16。"
    )
    lines.append("")

    # Figure captions + analysis placeholders filled with data-driven text
    fig_by_name = {p.name: p for p in figure_paths}

    def img(name: str) -> None:
        p = fig_by_name.get(name)
        if p is None:
            return
        rel = p.relative_to(REPORT_PATH.parent).as_posix()
        lines.append(f'<img src="{rel}" alt="{name}" width="780" />')
        lines.append("")

    def _r(a, b) -> str:
        if a is None or b is None or b == 0:
            return "—"
        return f"{a / b:.1f}×"

    # Pull a few numbers for per-figure prose (d=64 fp32)
    n_f = ms(IMPL_NAIVE, 32768, 64, "fp32", "forward_ms")
    p_f = ms(IMPL_PT, 32768, 64, "fp32", "forward_ms")
    t_f = ms(IMPL_TRITON, 32768, 64, "fp32", "forward_ms")
    n_b = ms(IMPL_NAIVE, 32768, 64, "fp32", "backward_ms")
    p_b = ms(IMPL_PT, 32768, 64, "fp32", "backward_ms")
    t_b = ms(IMPL_TRITON, 32768, 64, "fp32", "backward_ms")
    n_e = ms(IMPL_NAIVE, 32768, 64, "fp32", "e2e_ms")
    p_e = ms(IMPL_PT, 32768, 64, "fp32", "e2e_ms")
    t_e = ms(IMPL_TRITON, 32768, 64, "fp32", "e2e_ms")

    lines.append("## 2. 实验 A：主对比图与解析")
    lines.append("")
    lines.append("### 2.1 前向延迟 vs 序列长度")
    lines.append("")
    img("flash_bench_forward_vs_seq.png")
    lines.append(
        f"长序列上三条线应分开：naive attention 背 $S\\times S$ HBM 流量；"
        f"flash_pytorch 把大表切成 tile，显存压力下降，但仍是 Python 循环 + 多次小 kernel；"
        f"flash_triton 在片上融合 online Softmax。"
        f"锚点 $d=64$ fp32、$S=32768$：naive={_fmt_ms(n_f, False, False)} ms，"
        f"flash_pytorch={_fmt_ms(p_f, False, False)} ms，"
        f"flash_triton={_fmt_ms(t_f, False, False)} ms"
        f"（相对 naive：pt {_r(n_f, p_f)}，triton {_r(n_f, t_f)}）。"
        f"短 $S$ 上三条可能缠在一起——固定 launch / 编译开销尚未被 IO 优势淹没。"
    )
    lines.append("")
    lines.append("### 2.2 反向延迟 vs 序列长度")
    lines.append("")
    img("flash_bench_backward_vs_seq.png")
    lines.append(
        f"这里是相对旧实验变化最大的一张图：flash_triton 走分块反向，不应再与 naive 打平。"
        f"flash_pytorch 反向仍 dense 重算 $P$，预期贴近 naive attention。"
        f"锚点同上：naive={_fmt_ms(n_b, False, False)}，"
        f"pt={_fmt_ms(p_b, False, False)}，"
        f"triton={_fmt_ms(t_b, False, False)} ms"
        f"（pt/naive={_r(n_b, p_b)}，triton/naive={_r(n_b, t_b)}）。"
        f"若 pt 略慢于 naive，多半来自显式重算 $S/P$ 的路径，而不是「compile 让它变慢」这一笼统说法。"
    )
    lines.append("")
    lines.append("### 2.3 端到端延迟 vs 序列长度")
    lines.append("")
    img("flash_bench_e2e_vs_seq.png")
    lines.append(
        f"e2e = 前向 + 反向。flash_triton 若反向也融合，e2e 优势应接近前向优势；"
        f"flash_pytorch 则往往仍被 dense 反向拖住。"
        f"锚点：naive={_fmt_ms(n_e, False, False)}，"
        f"pt={_fmt_ms(p_e, False, False)}，"
        f"triton={_fmt_ms(t_e, False, False)} ms。"
    )
    lines.append("")
    lines.append("### 2.4 前向 / 反向 / 端到端 vs 隐藏维 $d$")
    lines.append("")
    img("flash_bench_forward_vs_d.png")
    lines.append(
        "固定若干 $S$，看 $d$ 增大时相对差距是否缩水："
        "两边都付 $O(S^2 d)$ matmul，naive 额外付与 $d$ 无关的 $S\\times S$ 存取；"
        "$d$ 越大，共同算力占比上升，Flash 相对优势常被稀释。"
        "Triton 大 $d$ 还会用更保守 tile（shared memory），launch 变多。"
    )
    lines.append("")
    img("flash_bench_backward_vs_d.png")
    lines.append(
        "反向 vs $d$：关注 flash_triton 是否仍系统性低于另两条；"
        "flash_pytorch 与 naive 是否保持同量级。"
    )
    lines.append("")
    img("flash_bench_e2e_vs_d.png")
    lines.append("端到端 vs $d$ 是前两张的合成；解读时对照同配置的 fwd/bwd，避免单独神话某一条曲线。")
    lines.append("")

    lines.append("## 3. 实验 B：tile 消融")
    lines.append("")
    img("flash_bench_tile_ablation.png")
    if tile:
        lines.append(
            "横轴为实际 $B_q$（与 $B_k$ 相同）；虚线 flash_pytorch，点线 flash_triton；颜色区分 $(S,d)$。"
            "预期：flash_pytorch 随 tile 增大而明显变快（循环次数↓、单次 GEMM 更大），"
            "但通常仍到不了同 tile 的 Triton；"
            "Triton 在过大 tile 上可能直接 OOM/编译失败（shared memory），"
            "此时 heuristic 的意义是「能跑且较快」而非「数学上最大 tile」。"
        )
        lines.append("")
        lines.append("| S | d | tile | flash_pytorch fwd/bwd/e2e | flash_triton fwd/bwd/e2e |")
        lines.append("|---:|---:|:-----|:--------------------------|:-------------------------|")
        # stable order
        labels_order = []
        seen = set()
        for r in tile:
            lab = r["tile_label"]
            if lab not in seen:
                seen.add(lab)
                labels_order.append(lab)
        for S, d in TILE_ABLATION_POINTS:
            # group by resolved tile_label
            labs = sorted(
                {
                    r["tile_label"]
                    for r in tile
                    if r["seq_len"] == S and r["d_model"] == d
                },
                key=lambda s: (0, int(s.split("x")[0])) if s[0].isdigit() else (1, s),
            )
            for lab in labs:
                pt = next((r for r in tile if r["seq_len"] == S and r["d_model"] == d and r["tile_label"] == lab and r["impl"] == IMPL_PT), None)
                tr = next((r for r in tile if r["seq_len"] == S and r["d_model"] == d and r["tile_label"] == lab and r["impl"] == IMPL_TRITON), None)

                def pack(r):
                    if r is None:
                        return "—"
                    return (
                        f"{_fmt_ms(r.get('forward_ms'), r.get('oom'), r.get('skipped'), r.get('error'))}/"
                        f"{_fmt_ms(r.get('backward_ms'), r.get('oom'), r.get('skipped'), r.get('error'))}/"
                        f"{_fmt_ms(r.get('e2e_ms'), r.get('oom'), r.get('skipped'), r.get('error'))}"
                    )

                lines.append(f"| {S} | {d} | {lab} | {pack(pt)} | {pack(tr)} |")
    else:
        lines.append("（本次未包含 tile 消融数据。）")
    lines.append("")

    lines.append("## 4. 总结论")
    lines.append("")
    lines.append(
        "1. **前向**：长序列上 flash_triton 应显著快于 naive attention；"
        "flash_pytorch 介于中间或偏慢——省的是 $S\\times S$ 显存形态，不是 Triton 级融合。"
    )
    lines.append(
        "2. **反向**：flash_pytorch（dense）≈ naive attention 量级；"
        "flash_triton（分块）才是「反向也 Flash」的对照。旧实验打平是因为当时测的是 dense 反向。"
    )
    lines.append(
        "3. **端到端**：由反向结构主导；只有 Triton 整条链路融合时，e2e 才会接近前向加速比。"
    )
    lines.append(
        "4. **$d$ 与 dtype**：大 $d$ 稀释相对 IO 优势；bf16 更利好带宽型的 naive，"
        "Triton 是否吃到 bf16 红利取决于 kernel，不能先验保证。"
    )
    lines.append(
        "5. **tile**：PyTorch 对小 tile 极敏感；加大 tile 是合法优化，但不能替代融合。"
        "主对比用相同 tile，是为了不把「分块大小」和「执行引擎」混为一谈；"
        "消融实验专门拆开看前者。"
    )
    lines.append("")

    lines.append("## 附录：主对比完整表（ms）")
    lines.append("")
    lines.append(
        "| S | d | dtype | naive fwd/bwd/e2e | flash_pytorch fwd/bwd/e2e | flash_triton fwd/bwd/e2e | tiles |"
    )
    lines.append("|---:|---:|:-----|:------------------|:--------------------------|:-------------------------|:------|")
    for dtype in ("fp32", "bf16"):
        for d_model in D_MODEL:
            for seq_len in SEQ_LENS:
                n = by.get((IMPL_NAIVE, seq_len, d_model, dtype))
                p = by.get((IMPL_PT, seq_len, d_model, dtype))
                t = by.get((IMPL_TRITON, seq_len, d_model, dtype))
                if n is None and p is None and t is None:
                    continue

                def pack(r):
                    if r is None:
                        return "—"
                    return (
                        f"{_fmt_ms(r.get('forward_ms'), bool(r.get('oom')), bool(r.get('skipped')), r.get('error'))}/"
                        f"{_fmt_ms(r.get('backward_ms'), bool(r.get('oom')), bool(r.get('skipped')), r.get('error'))}/"
                        f"{_fmt_ms(r.get('e2e_ms'), bool(r.get('oom')), bool(r.get('skipped')), r.get('error'))}"
                    )

                tile_s = "—"
                for src in (t, p):
                    if src and src.get("tile_label") not in (None, "n/a", "skipped"):
                        tile_s = src["tile_label"]
                        break
                lines.append(
                    f"| {seq_len} | {d_model} | {dtype} | {pack(n)} | {pack(p)} | {pack(t)} | {tile_s} |"
                )

    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT_PATH}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny grid for sanity check")
    p.add_argument("--from-json", type=Path, default=None, help="only plot/report from saved JSON")
    p.add_argument("--main-only", action="store_true")
    p.add_argument("--tile-only", action="store_true")
    args = p.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.from_json is not None:
        rows = json.loads(args.from_json.read_text())
    else:
        rows_dc: list[CellResult] = []
        if not args.tile_only:
            rows_dc.extend(run_main_sweep(smoke=args.smoke))
        if not args.main_only:
            rows_dc.extend(run_tile_ablation(smoke=args.smoke))
        rows = [asdict(r) for r in rows_dc]
        out_json = ARTIFACTS_DIR / "results.json"
        out_json.write_text(json.dumps(rows, indent=2))
        print(f"wrote {out_json}", flush=True)

    figs = make_figures(rows)
    write_report(rows, figs)


if __name__ == "__main__":
    main()
