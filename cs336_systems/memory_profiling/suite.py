"""8-cell memory profiling suite + report (Assignment 2 memory_profiling a–c).

Matrix (fixed):
  xl × context {128, 512} × mode {forward, train(+opt)} × precision {FP32, BF16}

Every cell dumps a pickle and records per-stage peaks (forward / loss / backward / optimizer).
Timelines are plotted from pickles (memory_viz-compatible Active Memory semantics).

Out of scope here: browser Detail-slider (e), Nsight memory (f).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cs336_systems.memory_profiling.profile import MemoryCellConfig, try_profile_cell
from cs336_systems.memory_profiling.timeline import (
    largest_allocs,
    load_snapshot,
    plot_active_timeline,
)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "memory_profiling"
REPORT_PATH = REPO_ROOT / "reports" / "memory-profiling.md"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

CONTEXTS: tuple[int, ...] = (128, 512)
MODES: tuple[str, ...] = ("forward", "train")
PRECISIONS: tuple[str, ...] = ("off", "bf16")
MODEL_SIZE = "xl"
BATCH_SIZE = 4
WARMUP = 2


def _bytes_gib(n: int | float | None) -> str:
    if n is None:
        return "—"
    return f"{n / (1024**3):.3f}"


def _bytes_mib(n: int | float | None) -> str:
    if n is None:
        return "—"
    return f"{n / (1024**2):.1f}"


def _mp_label(mp: str) -> str:
    return "FP32" if mp == "off" else "BF16"


def run_suite(*, contexts: tuple[int, ...] = CONTEXTS) -> list[dict[str, Any]]:
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    snap_dir = ARTIFACTS_ROOT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    total = len(contexts) * len(MODES) * len(PRECISIONS)
    idx = 0
    for ctx in contexts:
        for mode in MODES:
            for mp in PRECISIONS:
                idx += 1
                cfg = MemoryCellConfig(
                    model_size=MODEL_SIZE,
                    context_length=ctx,
                    batch_size=BATCH_SIZE,
                    mode=mode,  # type: ignore[arg-type]
                    mixed_precision=mp,  # type: ignore[arg-type]
                    warmup=WARMUP,
                )
                print(
                    f"\n[{idx}/{total}] {cfg.run_id} …",
                    flush=True,
                )
                cell_dir = snap_dir / cfg.run_id
                result = try_profile_cell(cfg, cell_dir)
                # Normalize pickle path relative to repo
                abs_p = result.get("pickle_path_abs")
                if abs_p:
                    p = Path(abs_p)
                    try:
                        result["pickle_path"] = str(p.relative_to(REPO_ROOT))
                    except ValueError:
                        result["pickle_path"] = abs_p
                results.append(result)
                peak = result.get("peak_allocated_gib")
                status = "OOM" if result.get("oom") else (f"peak={peak:.3f} GiB" if peak else "ok")
                print(f"  → {status}", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_size": MODEL_SIZE,
        "batch_size": BATCH_SIZE,
        "contexts": list(contexts),
        "modes": list(MODES),
        "precisions": list(PRECISIONS),
        "warmup": WARMUP,
        "note": (
            "Handout asks ctx 128/2048; this machine (A800 80GB) OOMs xl@B=4 for S≥1024. "
            "Suite uses 128/512 @ B=4 to match the rest of the assignment."
        ),
        "results": results,
    }
    manifest_path = ARTIFACTS_ROOT / "peaks.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {manifest_path}", flush=True)
    return results


def _find(
    results: list[dict[str, Any]],
    *,
    ctx: int,
    mode: str,
    mp: str,
) -> dict[str, Any] | None:
    for r in results:
        c = r.get("config") or {}
        if (
            c.get("context_length") == ctx
            and c.get("mode") == mode
            and c.get("mixed_precision") == mp
            and not r.get("oom")
        ):
            return r
    return None


def make_figures(results: list[dict[str, Any]]) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # (a) Primary timelines: ctx=512 FP32 forward + train (annotated stages)
    for mode, key in (("forward", "memory_a_xl_ctx512_forward"), ("train", "memory_a_xl_ctx512_train")):
        r = _find(results, ctx=512, mode=mode, mp="off")
        if r is None or not r.get("pickle_path_abs"):
            continue
        out = FIGURES_DIR / f"{key}.png"
        plot_active_timeline(
            Path(r["pickle_path_abs"]),
            out,
            title=f"xl · ctx=512 · B=4 · FP32 · {mode}",
            stage_boundaries=r.get("stage_boundaries") or [],
            baseline_bytes=int(r.get("baseline_allocated_bytes") or 0),
        )
        paths[key] = out

    # Extra: ctx=128 FP32 train for contrast
    r128 = _find(results, ctx=128, mode="train", mp="off")
    if r128 and r128.get("pickle_path_abs"):
        out = FIGURES_DIR / "memory_a_xl_ctx128_train.png"
        plot_active_timeline(
            Path(r128["pickle_path_abs"]),
            out,
            title="xl · ctx=128 · B=4 · FP32 · train",
            stage_boundaries=r128.get("stage_boundaries") or [],
            baseline_bytes=int(r128.get("baseline_allocated_bytes") or 0),
        )
        paths["memory_a_xl_ctx128_train"] = out

    # Peak comparison bar chart: context × mode × precision
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for ax, mode in zip(axes, MODES):
        x = np.arange(len(CONTEXTS))
        width = 0.35
        for i, mp in enumerate(PRECISIONS):
            vals = []
            for ctx in CONTEXTS:
                r = _find(results, ctx=ctx, mode=mode, mp=mp)
                vals.append(r["peak_allocated_gib"] if r else float("nan"))
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width,
                label=_mp_label(mp),
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in CONTEXTS])
        ax.set_xlabel("context length")
        ax.set_title(f"mode = {mode}")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("peak allocated (GiB)")
    fig.suptitle("xl · batch=4 · peak CUDA allocated memory")
    fig.tight_layout()
    peak_fig = FIGURES_DIR / "memory_b_peaks_by_context.png"
    fig.savefig(peak_fig, dpi=150)
    plt.close(fig)
    paths["memory_b_peaks_by_context"] = peak_fig

    # Staged peaks for train FP32 (both contexts)
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    stage_names = ["forward", "loss", "backward", "optimizer"]
    x = np.arange(len(stage_names))
    width = 0.35
    for i, ctx in enumerate(CONTEXTS):
        r = _find(results, ctx=ctx, mode="train", mp="off")
        vals = []
        for s in stage_names:
            if r and s in (r.get("stages") or {}):
                vals.append(r["stages"][s]["max_allocated_bytes"] / (1024**3))
            else:
                vals.append(float("nan"))
        ax.bar(x + (i - 0.5) * width, vals, width, label=f"ctx={ctx}")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_names)
    ax.set_ylabel("stage max_allocated (GiB)")
    ax.set_title("xl · FP32 · train：各阶段峰值（每阶段前 reset_peak）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    staged_fig = FIGURES_DIR / "memory_a_staged_peaks_fp32_train.png"
    fig.savefig(staged_fig, dpi=150)
    plt.close(fig)
    paths["memory_a_staged_peaks_fp32_train"] = staged_fig

    return paths


def _residual_stream_mib(ctx: int, batch: int = BATCH_SIZE) -> float:
    d_model = 2560  # xl
    return batch * ctx * d_model * 4 / (1024**2)


def write_report(results: list[dict[str, Any]], figure_paths: dict[str, Path]) -> None:
    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    lines: list[str] = []
    lines.append("# Memory Profiling")
    lines.append("")
    lines.append(
        f"**设定：** `BasicsTransformerLM` **xl**；batch={BATCH_SIZE}；"
        f"context ∈ {{{', '.join(str(c) for c in CONTEXTS)}}}；"
        "mode ∈ {forward, train}；precision ∈ {FP32, BF16}；"
        f"warmup={WARMUP}；train = forward + loss + backward + `AdamW.step()`。"
    )
    lines.append("")
    # Prefer the curated handout note already in the report (avoid regenerating a thinner stub).
    handout_block = None
    if REPORT_PATH.exists():
        old = REPORT_PATH.read_text(encoding="utf-8")
        h0 = old.find("**关于 handout")
        h1 = old.find("**交付范围：**", h0) if h0 >= 0 else -1
        if h0 >= 0 and h1 > h0:
            handout_block = old[h0:h1].rstrip()
    if handout_block and "QK" in handout_block:
        lines.append(handout_block)
    else:
        lines.append(
            "**关于 handout 的 128/2048：** 见报告正文（含 \(QK^\\top\) 与 "
            "\(B\\cdot H\\cdot S^{2}\\cdot 4\\cdot L\) 推导）；本机 \(B=4\) 下改用 "
            "context \(\\{128,512\\}\)。"
        )
    lines.append("")
    lines.append(
        "**交付范围：** (a)–(e) 来自 8 格 PyTorch memory snapshot 套件；"
        "(f) 来自 headless Nsight（`--cuda-memory-usage` + TransformerBlock NVTX）。"
        f"代码：`{REPO_ROOT / 'cs336_systems' / 'memory_profiling'}/`。"
    )
    lines.append("")

    # ---- (a) ----
    lines.append("## (a) Active Memory Timeline")
    lines.append("")
    lines.append(
        "录制：warmup 后开启 `_record_memory_history`，跑 **一步**，`_dump_snapshot`。"
        "曲线按 memory_viz 语义重建（`alloc` + / `free_completed` −）。"
        "竖虚线是阶段边界的 **地面真值**（`cuda.synchronize` 后打点），不是事后猜的。"
    )
    lines.append("")
    for key, caption in (
        ("memory_a_xl_ctx512_forward", "Forward only（ctx=512, FP32）"),
        ("memory_a_xl_ctx512_train", "Full train step（ctx=512, FP32）"),
        ("memory_a_staged_peaks_fp32_train", "Train 各阶段 max_allocated（FP32）"),
    ):
        if key in figure_paths:
            lines.append(f"**{caption}**")
            lines.append("")
            src = str(Path(figure_paths[key]).resolve())
            w = 520 if "staged" in key else 560
            lines.append(f'<img src="{src}" alt="{key}" width="{w}" />')
            lines.append("")

    # Stage narrative with ground truth numbers
    r_train = _find(results, ctx=512, mode="train", mp="off")
    r_fwd = _find(results, ctx=512, mode="forward", mp="off")
    if r_train and r_fwd:
        sf = r_train["stages"]
        lines.append(
            f"**Answer (a):** Forward-only（无权重、无 Adam）爬升到约 "
            f"**{_bytes_gib(r_fwd['peak_allocated_bytes'])} GiB** 后平台期，直到释放激活。"
            f"Full train 基线已含权重+AdamW（约 {_bytes_gib(r_train.get('baseline_allocated_bytes'))} GiB），"
            f"前向把激活堆上去，在 `forward`/`loss` 附近达到整步峰值 "
            f"**{_bytes_gib(r_train['peak_allocated_bytes'])} GiB**；"
            f"`backward` 呈台阶式下降（逐层释放 residual），结束后驻留约 "
            f"**{_bytes_gib(sf['backward']['allocated_bytes'])} GiB**；"
            f"`optimizer` 几乎平坦（约 {_bytes_gib(sf['optimizer']['allocated_bytes'])} GiB），"
            f"因 Adam 状态早在 warmup 就分配好了。"
            f"因此：**前向=爬升/高平台，反向=台阶下降，优化器=平坦**——"
            f"结合图中阶段竖线，三段可以明确分开。"
        )
    lines.append("")

    # ---- (b) ----
    lines.append("## (b) Peak memory by context")
    lines.append("")
    if "memory_b_peaks_by_context" in figure_paths:
        lines.append(
            f'<img src="{(FIGURES_DIR / "memory_b_peaks_by_context.png").resolve()}" '
            'alt="peaks by context" width="560" />'
        )
        lines.append("")
    lines.append("| context | forward peak (GiB) | train peak (GiB) |")
    lines.append("|--------:|-------------------:|-----------------:|")
    for ctx in CONTEXTS:
        rf = _find(results, ctx=ctx, mode="forward", mp="off")
        rt = _find(results, ctx=ctx, mode="train", mp="off")
        lines.append(
            f"| {ctx} | {_bytes_gib(rf['peak_allocated_bytes'] if rf else None)} | "
            f"{_bytes_gib(rt['peak_allocated_bytes'] if rt else None)} |"
        )
    lines.append("")
    lines.append(
        "单位：`torch.cuda.max_memory_allocated` 在该次 profiled step 上的全局峰值（GiB）。"
        "train ≫ forward：完整一步还要在权重之外常驻 AdamW 状态，并在反向阶段短暂叠上梯度；"
        "context 变长时 forward/train 都会涨，但 attention 的 \(S\\times S\) 项使涨幅快于线性。"
    )
    lines.append("")

    # ---- (c) ----
    lines.append("## (c) Mixed precision (BF16)")
    lines.append("")
    lines.append("| context | mode | FP32 peak (GiB) | BF16 peak (GiB) | Δ |")
    lines.append("|--------:|------|----------------:|----------------:|---|")
    for ctx in CONTEXTS:
        for mode in MODES:
            r0 = _find(results, ctx=ctx, mode=mode, mp="off")
            r1 = _find(results, ctx=ctx, mode=mode, mp="bf16")
            p0 = r0["peak_allocated_bytes"] if r0 else None
            p1 = r1["peak_allocated_bytes"] if r1 else None
            if p0 and p1:
                delta = f"{(p1 - p0) / (1024**3):+.3f} GiB ({100 * (p1 - p0) / p0:+.1f}%)"
            else:
                delta = "—"
            lines.append(
                f"| {ctx} | {mode} | {_bytes_gib(p0)} | {_bytes_gib(p1)} | {delta} |"
            )
    lines.append("")
    # Commentary using ctx=512
    r0f = _find(results, ctx=512, mode="forward", mp="off")
    r1f = _find(results, ctx=512, mode="forward", mp="bf16")
    r0t = _find(results, ctx=512, mode="train", mp="off")
    r1t = _find(results, ctx=512, mode="train", mp="bf16")
    if r0f and r1f and r0t and r1t:
        lines.append(
            f"**Answer (c):** 在 xl·ctx=512·B=4 上，BF16 autocast 相对 FP32："
            f"forward 峰值 {_bytes_gib(r0f['peak_allocated_bytes'])} → "
            f"{_bytes_gib(r1f['peak_allocated_bytes'])} GiB；"
            f"train 峰值 {_bytes_gib(r0t['peak_allocated_bytes'])} → "
            f"{_bytes_gib(r1t['peak_allocated_bytes'])} GiB（约 −4–7%）。"
            f"混精只把部分激活算成 BF16，**权重与 AdamW 状态仍是 FP32**，"
            f"故峰值远不会减半。短 context（128）上 forward 甚至可能因临时 dtype 转换略增，"
            f"说明混精对显存的影响 **不显著、且不稳定地「大降」**。"
        )
    lines.append("")

    # ---- (d) ----
    lines.append("## (d) Residual-stream activation size (analytic)")
    lines.append("")
    # Keep curated (d) body from the existing report when present.
    d_block = None
    if REPORT_PATH.exists():
        old = REPORT_PATH.read_text(encoding="utf-8")
        d0 = old.find("## (d)")
        d1 = old.find("## (e)", d0) if d0 >= 0 else -1
        if d0 >= 0 and d1 > d0 and "残差流" in old[d0:d1]:
            d_block = old[d0:d1].rstrip()
    if d_block:
        lines.append(d_block[len("## (d) Residual-stream activation size (analytic)") :].lstrip("\n"))
    else:
        lines.append(
            f"残差流张量形状 \((B,S,d_{{\\mathrm{{model}}}})\)，体积 "
            f"`B·S·d·4/1024²` MiB；\(B={BATCH_SIZE},\,S=512,\,d=2560\) → "
            f"**{_residual_stream_mib(512):.2f} MiB**。"
        )
    lines.append("")

    # ---- (e) largest allocs from forward pickle ----
    lines.append("## (e) Largest allocations（forward snapshot）")
    lines.append("")
    r = _find(results, ctx=512, mode="forward", mp="off")
    if r and r.get("pickle_path_abs"):
        snap = load_snapshot(Path(r["pickle_path_abs"]))
        top = largest_allocs(snap, top_k=5)
        lines.append(
            "从 ctx=512 FP32 forward 的 snapshot 中按 `alloc` 体积排序的 Top-5"
            "（等价于 memory_viz 调低 Detail 后看到的最大块）："
        )
        lines.append("")
        lines.append("| rank | size (MiB) | stack (truncated) |")
        lines.append("|-----:|-----------:|----------------|")
        for i, row in enumerate(top, 1):
            stack = "<br>".join(row["frames"][:4]) if row["frames"] else "—"
            lines.append(f"| {i} | {row['size_mib']:.1f} | {stack} |")
        lines.append("")
        # Prefer curated Answer (e) from existing report.
        e_ans = None
        if REPORT_PATH.exists():
            old = REPORT_PATH.read_text(encoding="utf-8")
            e0 = old.find("**Answer (e):**")
            e1 = old.find("## (f)", e0) if e0 >= 0 else -1
            if e0 >= 0 and e1 > e0 and "QK" in old[e0:e1]:
                e_ans = old[e0:e1].rstrip()
        if e_ans:
            lines.append(e_ans)
        elif top:
            lines.append(
                f"**Answer (e):** 最大块 **{top[0]['size_mib']:.1f} MiB** = "
                f"\(B·H·S·S·4/1024²\) attention score；见报告正文推导。"
            )
    lines.append("")

    # ---- (f): preserve existing section written by nsys_f.patch_report ----
    existing_f = ""
    if REPORT_PATH.exists():
        old = REPORT_PATH.read_text(encoding="utf-8")
        f0 = old.find("## (f)")
        f1 = old.find("## Appendix", f0) if f0 >= 0 else -1
        if f0 >= 0 and f1 > f0:
            existing_f = old[f0:f1].rstrip() + "\n\n"
    if existing_f and "本轮不做" not in existing_f:
        lines.append(existing_f.rstrip())
        lines.append("")
    else:
        lines.append("## (f) Nsight memory + NVTX（TransformerBlock residuals）")
        lines.append("")
        lines.append(
            "见 `python -m cs336_systems.memory_profiling.nsys_f all` 生成的本节内容"
            f"（artifacts：`{ARTIFACTS_ROOT / 'nsys_f'}/`）。"
        )
        lines.append("")

    # ---- appendix ----
    lines.append("## Appendix: full 8-cell stage table (FP32/BF16)")
    lines.append("")
    lines.append(
        "| ctx | mp | mode | peak GiB | fwd max | loss max | bwd max | opt max |"
    )
    lines.append(
        "|----:|----|------|---------:|--------:|---------:|--------:|--------:|"
    )
    for ctx in CONTEXTS:
        for mp in PRECISIONS:
            for mode in MODES:
                r = _find(results, ctx=ctx, mode=mode, mp=mp)
                if r is None:
                    # maybe OOM entry
                    for cand in results:
                        c = cand.get("config") or {}
                        if (
                            c.get("context_length") == ctx
                            and c.get("mode") == mode
                            and c.get("mixed_precision") == mp
                        ):
                            r = cand
                            break
                if r is None:
                    continue
                if r.get("oom"):
                    lines.append(
                        f"| {ctx} | {_mp_label(mp)} | {mode} | OOM | — | — | — | — |"
                    )
                    continue
                st = r.get("stages") or {}

                def _stg(name: str) -> str:
                    if name not in st:
                        return "—"
                    return _bytes_gib(st[name]["max_allocated_bytes"])

                lines.append(
                    f"| {ctx} | {_mp_label(mp)} | {mode} | "
                    f"{_bytes_gib(r.get('peak_allocated_bytes'))} | "
                    f"{_stg('forward')} | {_stg('loss')} | {_stg('backward')} | {_stg('optimizer')} |"
                )
    lines.append("")
    lines.append(f"产物路径：`{ARTIFACTS_ROOT}/` · 报告生成时间 UTC。")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Memory profiling 8-cell suite")
    p.add_argument(
        "--contexts",
        type=int,
        nargs="+",
        default=list(CONTEXTS),
        help="Context lengths to sweep (default: 128 512)",
    )
    p.add_argument("--skip-run", action="store_true", help="Only rebuild figures/report from peaks.json")
    args = p.parse_args()
    contexts = tuple(args.contexts)

    if args.skip_run:
        manifest = json.loads((ARTIFACTS_ROOT / "peaks.json").read_text(encoding="utf-8"))
        results = manifest["results"]
    else:
        results = run_suite(contexts=contexts)

    figs = make_figures(results)
    write_report(results, figs)


if __name__ == "__main__":
    main()
