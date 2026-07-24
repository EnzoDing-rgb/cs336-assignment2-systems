"""Nsight Systems parts (c)(d)(e): non-GEMM forward kernels, full train step, attn softmax vs matmul.

One suite profiles medium × {256,512,1024} with:
  outer NVTX ``train_step`` = zero_grad → forward → loss → backward → AdamW.step
  inner NVTX ``forward``
  attention SDPA patched with ``attn_matmul`` / ``attn_softmax``

(c) reuses existing (a)(b) forward kern tables (no re-profile).
Report appends to reports/nsys-profile.md; new figures under reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
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
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_WARMUP,
    MODEL_SIZE_PRESETS,
    BenchmarkConfig,
    _format_seconds,
    _sync,
    build_model,
    make_batch,
)
from cs336_systems.nsys_profile.ab import (
    ARTIFACTS_ROOT,
    FIGURES_DIR,
    NVTX_FORWARD,
    REPO_ROOT,
    REPORT_PATH,
    SUITE_MANIFEST,
    _is_elementwise,
    _is_gemm,
    _ns_to_s,
    _nsys_bin,
    _read_csv,
    _rel,
    _row_get,
    _short_kernel_label,
    export_stats,
    load_range_kernel_totals,
    md_img,
    share_breakdown,
)
from cs336_systems.nsys_profile.attn_nvtx import (
    NVTX_ATTN_MATMUL,
    NVTX_ATTN_SOFTMAX,
    install_attn_nvtx,
    uninstall_attn_nvtx,
)

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CDE_ARTIFACTS = ARTIFACTS_ROOT / "cde"
CDE_MANIFEST = CDE_ARTIFACTS / "suite_manifest.json"
CDE_SIZES: tuple[str, ...] = ("medium",)
CDE_CONTEXTS: tuple[int, ...] = (256, 512, 1024)
FOCUS_SIZE = "medium"
FOCUS_CTX = 512

NVTX_TRAIN = "train_step"
NVTX_WARMUP = "warmup"

FIGURE_C = FIGURES_DIR / "nsys_c_nongemm_forward.png"
FIGURE_D = FIGURES_DIR / "nsys_d_matmul_fraction_forward_vs_train.png"
FIGURE_E = FIGURES_DIR / "nsys_e_attn_softmax_vs_matmul.png"


@dataclass
class CdeCellResult:
    model_size: str
    context_length: int
    train_step_mean_s: float | None
    forward_mean_s: float | None
    attn_matmul_total_s: float | None
    attn_softmax_total_s: float | None
    pct_gemm_forward: float | None
    pct_gemm_train: float | None
    pct_other_forward: float | None
    pct_other_train: float | None
    flops_attn_matmul: float | None
    flops_attn_softmax: float | None
    oom: bool = False
    artifact_dir: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CdeCellResult:
        return cls(**d)


def _cfg(size: str, context: int, warmup: int = DEFAULT_WARMUP, steps: int = DEFAULT_STEPS) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_size=size,
        mode="train",
        context_length=context,
        warmup=warmup,
        steps=steps,
        seed=DEFAULT_SEED,
        device="cuda",
    )


def _run_train_step_nvtx(
    model: torch.nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    label_train: str,
    label_forward: str,
) -> None:
    with nvtx.range(label_train):
        optimizer.zero_grad(set_to_none=True)
        with nvtx.range(label_forward):
            logits = model(x)
            _sync(device)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        _sync(device)


def run_train_workload(cfg: BenchmarkConfig) -> None:
    """Full AdamW train step under nsys; attention SDPA patched with fine NVTX."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    install_attn_nvtx()
    try:
        device = torch.device("cuda")
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.empty_cache()

        print(
            f"[cde-workload] size={cfg.model_size} ctx={cfg.context_length} "
            f"warmup={cfg.warmup} steps={cfg.steps}",
            flush=True,
        )
        model = build_model(cfg, device)
        model.train()
        optimizer = AdamW(model.parameters())
        x, y = make_batch(cfg, device)

        for i in range(cfg.warmup):
            with nvtx.range(NVTX_WARMUP):
                _run_train_step_nvtx(
                    model,
                    optimizer,
                    x,
                    y,
                    device,
                    label_train=f"{NVTX_WARMUP}_train",
                    label_forward=f"{NVTX_WARMUP}_forward",
                )
            print(f"[cde-workload] warmup {i + 1}/{cfg.warmup}", flush=True)

        for i in range(cfg.steps):
            _run_train_step_nvtx(
                model,
                optimizer,
                x,
                y,
                device,
                label_train=NVTX_TRAIN,
                label_forward=NVTX_FORWARD,
            )
            print(f"[cde-workload] measure {i + 1}/{cfg.steps}", flush=True)

        print("[cde-workload] done", flush=True)
    finally:
        uninstall_attn_nvtx()


def profile_train_with_nsys(cfg: BenchmarkConfig, out_prefix: Path) -> Path:
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
        "--force-overwrite=true",
        "-o",
        str(out_prefix),
        sys.executable,
        "-m",
        "cs336_systems.nsys_profile.cde",
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if proc.returncode != 0:
        raise RuntimeError(f"nsys profile failed with exit code {proc.returncode}")
    if not rep.exists():
        raise RuntimeError(f"expected report missing: {rep}")
    return rep


def parse_nvtx_avg(nvtxsum_csv: Path, range_name: str) -> tuple[float, int]:
    rows = _read_csv(nvtxsum_csv)
    for row in rows:
        if _row_get(row, "Range", "NVTX Range").strip() == range_name:
            avg = _row_get(row, "Avg", "Average")
            inst = int(float(_row_get(row, "Instances", "NVTX Inst") or 0))
            return _ns_to_s(avg), inst
    raise RuntimeError(f"NVTX range {range_name!r} not found in {nvtxsum_csv}")


def parse_nvtx_total(nvtxsum_csv: Path, range_name: str) -> float:
    rows = _read_csv(nvtxsum_csv)
    for row in rows:
        if _row_get(row, "Range", "NVTX Range").strip() == range_name:
            return _ns_to_s(_row_get(row, "Total Time", "Total"))
    raise RuntimeError(f"NVTX range {range_name!r} not found in {nvtxsum_csv}")


def attn_flops_estimate(cfg: BenchmarkConfig) -> dict[str, float]:
    """Analytical FLOPs for one forward over all layers (SDPA only: QK/AV vs softmax).

    Matmul (standard 2mnk for (m,k)@(k,n)):
      QKᵀ: B·H · 2·S·S·d_k
      scale by 1/√d_k: B·H·S·S   (elementwise)
      AV:   B·H · 2·S·S·d_v     (d_v = d_k here)

    Softmax on scores of shape (B, H, S, S), last dim = S
    (matches cs336_basics.nn_utils.softmax):
      per row of length S:
        max ≈ S-1 comps, sub S, exp S, sum S-1, div S  →  ≈ 5S - 2
      total ≈ B·H·S·(5S - 2)

    Mask ``where`` is ignored (select / store -inf, not arithmetic FLOPs).
    ``exp`` counted as 1 FLOP each (common ML convention; hardware cost is higher).
    """
    preset = MODEL_SIZE_PRESETS[cfg.model_size]
    b = cfg.batch_size
    s = cfg.context_length
    h = preset["num_heads"]
    d_k = preset["d_model"] // h
    d_v = d_k
    n_layers = preset["num_layers"]

    qk = b * h * 2 * s * s * d_k
    scale = b * h * s * s
    av = b * h * 2 * s * s * d_v
    matmul_per_layer = qk + scale + av

    # Exact row cost 5S-2; for reporting keep both exact and the ~5S² shorthand.
    soft_per_layer = b * h * s * (5 * s - 2)

    return {
        "flops_qk": float(n_layers * qk),
        "flops_scale": float(n_layers * scale),
        "flops_av": float(n_layers * av),
        "flops_attn_matmul": float(n_layers * matmul_per_layer),
        "flops_attn_softmax": float(n_layers * soft_per_layer),
        "d_k": float(d_k),
        "B": float(b),
        "H": float(h),
        "S": float(s),
        "L": float(n_layers),
    }


def _gemm_fraction(ranked: list[tuple[str, float]]) -> tuple[float, float]:
    total = sum(t for _, t in ranked)
    if total <= 0:
        return 0.0, 0.0
    gemm = sum(t for n, t in ranked if _is_gemm(n))
    return 100.0 * gemm / total, 100.0 * (total - gemm) / total


def _is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda oom" in msg or isinstance(err, torch.cuda.OutOfMemoryError)


def run_cde_cell(size: str, context: int, cell_dir: Path) -> CdeCellResult:
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(size, context)
    try:
        print(f"[cde-cell] nsys train profile {size} ctx={context}", flush=True)
        rep = profile_train_with_nsys(cfg, cell_dir / "profile")
        stats = export_stats(rep, cell_dir)
        train_mean, _ = parse_nvtx_avg(stats["nvtxsum"], NVTX_TRAIN)
        fwd_mean, _ = parse_nvtx_avg(stats["nvtxsum"], NVTX_FORWARD)
        attn_mm = parse_nvtx_total(stats["nvtxsum"], NVTX_ATTN_MATMUL)
        attn_sm = parse_nvtx_total(stats["nvtxsum"], NVTX_ATTN_SOFTMAX)
        gemm_f, other_f = _gemm_fraction(load_range_kernel_totals(stats["nvtxkernsum"], NVTX_FORWARD))
        gemm_t, other_t = _gemm_fraction(load_range_kernel_totals(stats["nvtxkernsum"], NVTX_TRAIN))
        flops = attn_flops_estimate(cfg)
        flops_mm = flops["flops_attn_matmul"]
        flops_sm = flops["flops_attn_softmax"]
        result = CdeCellResult(
            model_size=size,
            context_length=context,
            train_step_mean_s=train_mean,
            forward_mean_s=fwd_mean,
            attn_matmul_total_s=attn_mm,
            attn_softmax_total_s=attn_sm,
            pct_gemm_forward=gemm_f,
            pct_gemm_train=gemm_t,
            pct_other_forward=other_f,
            pct_other_train=other_t,
            flops_attn_matmul=flops_mm,
            flops_attn_softmax=flops_sm,
            oom=False,
            artifact_dir=_rel(cell_dir),
        )
    except Exception as err:  # noqa: BLE001
        if _is_oom(err):
            torch.cuda.empty_cache()
            result = CdeCellResult(
                model_size=size,
                context_length=context,
                train_step_mean_s=None,
                forward_mean_s=None,
                attn_matmul_total_s=None,
                attn_softmax_total_s=None,
                pct_gemm_forward=None,
                pct_gemm_train=None,
                pct_other_forward=None,
                pct_other_train=None,
                flops_attn_matmul=None,
                flops_attn_softmax=None,
                oom=True,
                artifact_dir=_rel(cell_dir),
                notes=f"OOM: {err}",
            )
        else:
            result = CdeCellResult(
                model_size=size,
                context_length=context,
                train_step_mean_s=None,
                forward_mean_s=None,
                attn_matmul_total_s=None,
                attn_softmax_total_s=None,
                pct_gemm_forward=None,
                pct_gemm_train=None,
                pct_other_forward=None,
                pct_other_train=None,
                flops_attn_matmul=None,
                flops_attn_softmax=None,
                oom=False,
                artifact_dir=_rel(cell_dir),
                notes=f"failed: {err}",
            )
    (cell_dir / "cell_summary.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _ab_focus_kern_csv() -> Path | None:
    """Locate medium@512 kern CSV from the (a)(b) suite."""
    if SUITE_MANIFEST.exists():
        man = json.loads(SUITE_MANIFEST.read_text(encoding="utf-8"))
        for c in man.get("cells", []):
            if c.get("model_size") == FOCUS_SIZE and c.get("context_length") == FOCUS_CTX and not c.get("oom"):
                ad = c.get("artifact_dir")
                if ad:
                    base = Path(ad) if Path(ad).is_absolute() else REPO_ROOT / ad
                    matches = sorted(base.glob("stats*nvtx_kern_sum.csv"))
                    if matches:
                        return matches[-1]
    # fallback scan
    for p in sorted((ARTIFACTS_ROOT).glob("suite_*/medium_ctx512/stats*nvtx_kern_sum.csv")):
        return p
    return None


def plot_part_c(out_path: Path) -> dict[str, Any] | None:
    csv_path = _ab_focus_kern_csv()
    if csv_path is None:
        print("[plot] skip (c): no (a)(b) kern csv", flush=True)
        return None
    ranked = load_range_kernel_totals(csv_path, NVTX_FORWARD)
    bd = share_breakdown(ranked)
    nongemm = [(n, t, 100.0 * t / bd["total_s"]) for n, t in ranked if not _is_gemm(n)]
    nongemm = nongemm[:8]
    if not nongemm:
        print("[plot] skip (c): no non-gemm kernels", flush=True)
        return None

    labels = [_short_kernel_label(n) for n, _, _ in nongemm][::-1]
    pcts = [p for _, _, p in nongemm][::-1]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.barh(labels, pcts, color="#e45756")
    ax.set_xlabel("占 forward 全部 CUDA 内核时间的 %")
    ax.set_title(f"Part (c): {FOCUS_SIZE}@ctx{FOCUS_CTX} forward 中非矩阵乘内核")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path.name}", flush=True)
    return {
        "nongemm": [{"kernel": n, "total_s": t, "pct": p} for n, t, p in nongemm],
        "pct_gemm": bd["pct_gemm"],
        "pct_elementwise": bd["pct_elementwise"],
        "total_s": bd["total_s"],
    }


def plot_part_d(cells: list[CdeCellResult], out_path: Path) -> None:
    ok = [c for c in cells if not c.oom and c.pct_gemm_forward is not None]
    if not ok:
        print(f"[plot] skip {out_path.name}: no data", flush=True)
        return
    labels = [f"ctx={c.context_length}" for c in ok]
    xs = list(range(len(ok)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fwd = [c.pct_gemm_forward for c in ok]
    trn = [c.pct_gemm_train for c in ok]
    ax.bar([x - width / 2 for x in xs], fwd, width, label="forward only（matmul %）", color="#4c78a8")
    ax.bar([x + width / 2 for x in xs], trn, width, label="完整 train_step（matmul %）", color="#f58518")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GEMM 占该范围全部 CUDA 内核时间的 %")
    ax.set_ylim(0, 100)
    ax.set_title(f"Part (d): {FOCUS_SIZE} — matmul 占比：前向 vs 完整训练步")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot] wrote {out_path.name}", flush=True)


def plot_part_e(cells: list[CdeCellResult], out_path: Path) -> None:
    """Side-by-side: runtime (close) vs FLOPs (softmax tiny on linear scale)."""
    ok = [c for c in cells if not c.oom and c.attn_matmul_total_s and c.attn_softmax_total_s]
    if not ok:
        print(f"[plot] skip {out_path.name}: no data", flush=True)
        return

    # Prefer freshly computed analytical FLOPs (includes scale term; exact softmax 5S-2).
    for c in ok:
        fl = attn_flops_estimate(_cfg(c.model_size, c.context_length))
        c.flops_attn_matmul = fl["flops_attn_matmul"]
        c.flops_attn_softmax = fl["flops_attn_softmax"]

    labels = [f"ctx={c.context_length}" for c in ok]
    xs = list(range(len(ok)))
    width = 0.36
    mm_ms = [c.attn_matmul_total_s * 1e3 for c in ok]
    sm_ms = [c.attn_softmax_total_s * 1e3 for c in ok]
    # One forward's analytical FLOPs × measure steps (10), to match cumulative NVTX totals.
    steps = DEFAULT_STEPS
    mm_gflops = [(c.flops_attn_matmul or 0.0) * steps / 1e9 for c in ok]
    sm_gflops = [(c.flops_attn_softmax or 0.0) * steps / 1e9 for c in ok]

    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(10.5, 4.6))

    b1 = ax_t.bar([x - width / 2 for x in xs], mm_ms, width, label="attn_matmul", color="#4c78a8")
    b2 = ax_t.bar([x + width / 2 for x in xs], sm_ms, width, label="attn_softmax", color="#54a24b")
    ax_t.set_xticks(xs)
    ax_t.set_xticklabels(labels)
    ax_t.set_ylabel("累计时间（10 次 forward 合计，ms）")
    ax_t.set_title("运行时间（线性轴）\n两者接近")
    ax_t.legend(loc="upper left", fontsize=8)
    ax_t.grid(axis="y", alpha=0.3)
    for cell, x, a, b in zip(ok, xs, mm_ms, sm_ms):
        ax_t.text(x, max(a, b) * 1.02, f"soft/mm={b / a:.2f}", ha="center", fontsize=8)

    ax_f.bar([x - width / 2 for x in xs], mm_gflops, width, label="attn_matmul", color="#4c78a8")
    ax_f.bar([x + width / 2 for x in xs], sm_gflops, width, label="attn_softmax", color="#54a24b")
    ax_f.set_xticks(xs)
    ax_f.set_xticklabels(labels)
    ax_f.set_ylabel("解析 FLOPs ×10 步（GFLOP，线性轴）")
    ax_f.set_title("算力账本（线性轴）\nsoftmax 几乎看不见")
    ax_f.legend(loc="upper left", fontsize=8)
    ax_f.grid(axis="y", alpha=0.3)
    for cell, x, a, b in zip(ok, xs, mm_gflops, sm_gflops):
        ax_f.text(x, a * 1.02, f"soft/mm={b / a:.3f}", ha="center", fontsize=8)

    fig.suptitle(
        f"Part (e): {FOCUS_SIZE} — attention 内 softmax vs matmul\n"
        "左：实测时间接近；右：按 FLOPs 算 softmax 只占 matmul 的 ~2%",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_path.name}", flush=True)
    _ = (b1, b2)  # silence unused if linty


def append_cde_report(
    cells: list[CdeCellResult],
    part_c: dict[str, Any] | None,
    path: Path = REPORT_PATH,
) -> None:
    focus = next((c for c in cells if c.model_size == FOCUS_SIZE and c.context_length == FOCUS_CTX and not c.oom), None)
    lines: list[str] = [
        "",
        "---",
        "",
        "## (c) Non-matmul kernels in the forward pass",
        "",
        md_img(f"figures/{FIGURE_C.name}", alt="part c", width=480),
        "",
    ]
    if part_c:
        top_names = []
        for item in part_c["nongemm"][:5]:
            top_names.append(f"`{_short_kernel_label(item['kernel'])}` ({item['pct']:.1f}%)")
        lines += [
            f"**Answer (c):** On `{FOCUS_SIZE}` context {FOCUS_CTX}, besides GEMMs "
            f"(≈{part_c['pct_gemm']:.1f}% of forward CUDA time), non-trivial time goes to "
            f"elementwise kernels (≈{part_c['pct_elementwise']:.1f}%, mainly `MulFunctor` / binary mul) "
            f"and other ATen helpers such as {', '.join(top_names[:3])}.",
            "",
        ]
    else:
        lines += ["**Answer (c):** See figure; (a)(b) kern table unavailable for narrative.", ""]

    lines += [
        "## (d) Full training step vs forward-only: matmul fraction",
        "",
        md_img(f"figures/{FIGURE_D.name}", alt="part d", width=520),
        "",
        "| size | context | GEMM % (forward) | GEMM % (train_step) | other % (forward) | other % (train) | train_step mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for size in CDE_SIZES:
        for ctx in CDE_CONTEXTS:
            cell = next((c for c in cells if c.model_size == size and c.context_length == ctx), None)
            if cell is None:
                lines.append(f"| {size} | {ctx} | — | — | — | — | — |")
            elif cell.oom:
                lines.append(f"| {size} | {ctx} | OOM | OOM | OOM | OOM | OOM |")
            else:
                lines.append(
                    f"| {size} | {ctx} | {cell.pct_gemm_forward:.1f}% | {cell.pct_gemm_train:.1f}% | "
                    f"{cell.pct_other_forward:.1f}% | {cell.pct_other_train:.1f}% | "
                    f"{_format_seconds(cell.train_step_mean_s or float('nan'))} |"
                )

    if focus and focus.pct_gemm_forward is not None and focus.pct_gemm_train is not None:
        delta = focus.pct_gemm_train - focus.pct_gemm_forward
        lines += [
            "",
            f"**Answer (d):** On `{FOCUS_SIZE}` context {FOCUS_CTX}, GEMM's share of CUDA kernel time "
            f"drops from {focus.pct_gemm_forward:.1f}% in the nested `forward` range to "
            f"{focus.pct_gemm_train:.1f}% over the full `train_step` (Δ {delta:+.1f} pp), because "
            f"backward plus AdamW add substantial non-GEMM (and some GEMM) work; other kernels rise to "
            f"{focus.pct_other_train:.1f}% of train-step CUDA time.",
            "",
        ]
    else:
        lines += ["", "**Answer (d):** See table above.", ""]

    lines += [
        "## (e) Softmax vs matmul inside self-attention (forward)",
        "",
        "左图：NVTX 实测累计时间（线性轴）；右图：同一次 forward 的解析 FLOPs×10 步（**同样线性轴**）。"
        "右边 softmax 柱几乎贴地，左边两根柱却差不多高——这就是本题要你看见的反差。",
        "",
        md_img(f"figures/{FIGURE_E.name}", alt="part e", width=720),
        "",
        "| size | context | attn_matmul total | attn_softmax total | time ratio soft/mm | FLOPs matmul (1 fwd) | FLOPs softmax (1 fwd) | FLOPs ratio soft/mm |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in CDE_SIZES:
        for ctx in CDE_CONTEXTS:
            cell = next((c for c in cells if c.model_size == size and c.context_length == ctx), None)
            if cell is None or cell.oom or cell.attn_matmul_total_s is None:
                lines.append(f"| {size} | {ctx} | — | — | — | — | — | — |")
                continue
            tr = cell.attn_softmax_total_s / cell.attn_matmul_total_s if cell.attn_matmul_total_s else float("nan")
            fr = (
                cell.flops_attn_softmax / cell.flops_attn_matmul
                if cell.flops_attn_matmul
                else float("nan")
            )
            lines.append(
                f"| {size} | {ctx} | {_format_seconds(cell.attn_matmul_total_s)} | "
                f"{_format_seconds(cell.attn_softmax_total_s)} | {tr:.3f} | "
                f"{cell.flops_attn_matmul:.2e} | {cell.flops_attn_softmax:.2e} | {fr:.4f} |"
            )

    if focus and focus.attn_matmul_total_s and focus.attn_softmax_total_s:
        fl = attn_flops_estimate(_cfg(FOCUS_SIZE, FOCUS_CTX))
        focus.flops_attn_matmul = fl["flops_attn_matmul"]
        focus.flops_attn_softmax = fl["flops_attn_softmax"]
        tr = focus.attn_softmax_total_s / focus.attn_matmul_total_s
        fr = focus.flops_attn_softmax / focus.flops_attn_matmul
        mm_ms = focus.attn_matmul_total_s * 1e3
        sm_ms = focus.attn_softmax_total_s * 1e3
        expected_sm_ms = mm_ms * fr
        slowdown = sm_ms / expected_sm_ms if expected_sm_ms > 0 else float("nan")
        preset = MODEL_SIZE_PRESETS[FOCUS_SIZE]
        d_k = preset["d_model"] // preset["num_heads"]
        b, h, s, ell = 4, preset["num_heads"], FOCUS_CTX, preset["num_layers"]
        lines += [
            "",
            f"### 详解：以 `{FOCUS_SIZE}`、上下文 {FOCUS_CTX} 为例",
            "",
            "**我们到底在比什么。**  "
            "只比 self-attention 里 `scaled_dot_product_attention` 这一段，不把 Q/K/V 线性投影算进去："
            f"NVTX `{NVTX_ATTN_MATMUL}` = `QKᵀ`、除以 `√d_k`、以及 `AV`；"
            f"`{NVTX_ATTN_SOFTMAX}` = `cs336_basics.nn_utils.softmax`（max → 减 max → exp → sum → 除法）。"
            f"上表时间为 **10 次正式 forward** 的 NVTX Total；FLOPs 是 **单次 forward、全部 {ell} 层** 的解析估计。",
            "",
            "**实测时间。**  "
            f"`attn_matmul` 累计 **{mm_ms:.1f} ms**，`attn_softmax` 累计 **{sm_ms:.1f} ms**，"
            f"比值 soft/mm = **{tr:.2f}**。",
            "",
            "#### FLOPs 怎么一笔笔加出来的",
            "",
            f"形状约定（`medium`）：**B={b}，H={h}，S={s}，d_k=d_v={d_k}，L={ell}**。"
            "矩阵乘用教材惯例：`(m×k)·(k×n)` 计 **2·m·k·n** 次浮点运算（每次乘加算 2）。"
            "`exp` 按 ML 文献惯例计 **1 FLOP / 元素**（硬件上更贵，见下）。因果 `where` 掩码不计算术 FLOPs。",
            "",
            "**1）矩阵乘侧（计入 `attn_matmul`）——每一层：**",
            "",
            f"- **QKᵀ**：每个 head 做 `(S×d_k)·(d_k×S) → (S×S)`，"
            f"FLOPs = `2·S·d_k·S`；共 B·H 个 head → "
            f"`2·B·H·S²·d_k` = `2·{b}·{h}·{s}²·{d_k}` = **{fl['flops_qk']/ell:,.0f}**。",
            f"- **缩放** `scores / √d_k`：对 `B·H·S·S` 个元素各做一次除法 → "
            f"`B·H·S²` = **{fl['flops_scale']/ell:,.0f}**（相对 GEMM 是低阶项，但代码里确实在算）。",
            f"- **AV**：`(S×S)·(S×d_v) → (S×d_v)`，"
            f"FLOPs = `2·S·S·d_v`；×B·H → "
            f"`2·B·H·S²·d_v` = **{fl['flops_av']/ell:,.0f}**。",
            f"- **一层合计** ≈ **{fl['flops_attn_matmul']/ell:,.0f}**；"
            f"**L={ell} 层合计（一次 forward）** = **{fl['flops_attn_matmul']:.6e}**。",
            "",
            "**2）Softmax 侧（计入 `attn_softmax`）——对照源码逐步数：**",
            "",
            "```python",
            "rescaled = x - max(x)      # 沿最后一维（key 维，长度 S）",
            "exps = exp(rescaled)",
            "return exps / sum(exps)",
            "```",
            "",
            f"分数张量形状 `(B, H, S, S)`：共有 **B·H·S = {b}·{h}·{s} = {b*h*s}** 行，"
            "每一行是长度 **S** 的向量，对这一行：",
            "",
            f"| 步骤 | 运算 | 每行大约几次浮点/比较 |",
            f"|---|---|---:|",
            f"| max | 找最大值 | S−1 ≈ {s - 1} |",
            f"| 减 max | 元素减 | S = {s} |",
            f"| exp | 逐元素 exp | S = {s}（各计 1 FLOP） |",
            f"| sum | 求和 | S−1 ≈ {s - 1} |",
            f"| 除法 | 归一化 | S = {s} |",
            f"| **一行合计** | | **5S−2 = {5 * s - 2}** |",
            "",
            f"一层：`B·H·S·(5S−2)` = **{fl['flops_attn_softmax']/ell:,.0f}**；"
            f"**L 层一次 forward** = **{fl['flops_attn_softmax']:.6e}**。"
            f"（若用更粗的 `≈5·B·H·S²`，相对误差只有 O(1/S)，S={s} 时可忽略。）",
            "",
            f"**3）比值。** soft/mm = "
            f"{fl['flops_attn_softmax']:.3e} / {fl['flops_attn_matmul']:.3e} ≈ **{fr:.4f}**。"
            f"主阶近似：soft/mm ≈ `(5S) / (4·S·d_k + S)` ≈ `5 / (4·d_k)` = `5/(4·{d_k})` ≈ "
            f"{5 / (4 * d_k):.4f}（缩放项相对 4·d_k 很小）。",
            "",
            "**反差有多大。**  "
            f"若运行时间真按 FLOPs 成比例，softmax 相对 matmul 的 {mm_ms:.1f} ms 应只占约 "
            f"**{expected_sm_ms:.1f} ms**；实测却是 **{sm_ms:.1f} ms**，约是「按 FLOPs 预言」的 "
            f"**{slowdown:.0f}×**。算力账上 softmax ≈ matmul 的 **{100 * fr:.1f}%**，墙上时钟却到 **{100 * tr:.0f}%**。",
            "",
            "**为什么时间远比 FLOPs 接近？**  "
            "1）**算术强度**：GEMM 每个输出吃掉 `d_k` 次乘加，易喂满 Tensor Core；"
            "softmax 对 `S×S` 矩阵几乎是读/写带宽活，算得少、搬得多。"
            "2）**`exp` 被低估**：账本里 `exp` 只算 1 FLOP，真实延迟远高于一次 add。"
            "3）**内核形态**：matmul 并成大 SGEMM；softmax 是一串短 elementwise/reduce，launch 多、利用率差。"
            "4）三个 context 上 soft/mm **时间比**仍在 0.64–0.84，而 **FLOPs 比**钉在 ~0.02。",
            "",
            f"**Answer (e):** On `{FOCUS_SIZE}` ctx {FOCUS_CTX}, attention softmax takes "
            f"{_format_seconds(focus.attn_softmax_total_s)} vs matmul {_format_seconds(focus.attn_matmul_total_s)} "
            f"(time ratio {tr:.2f}), but analytical FLOPs give soft/mm ≈ {fr:.4f}; "
            f"softmax runs ~{slowdown:.0f}× longer than a FLOP-proportional prediction from the matmul time, "
            f"because it is bandwidth- and launch-bound rather than compute-bound.",
            "",
        ]
    else:
        lines += ["", "**Answer (e):** See table above.", ""]

    # Refresh analytical FLOPs in the table rows too.
    refreshed: list[str] = []
    for line in lines:
        refreshed.append(line)
    # Rebuild e-table FLOPs columns by rewriting after the header — done above via focus;
    # also fix per-cell table entries:
    for i, line in enumerate(lines):
        if line.startswith("| medium |"):
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 8 and parts[0] == "medium":
                try:
                    ctx = int(parts[1])
                except ValueError:
                    continue
                cell = next((c for c in cells if c.model_size == "medium" and c.context_length == ctx), None)
                if cell is None or cell.oom or cell.attn_matmul_total_s is None:
                    continue
                flc = attn_flops_estimate(_cfg("medium", ctx))
                cell.flops_attn_matmul = flc["flops_attn_matmul"]
                cell.flops_attn_softmax = flc["flops_attn_softmax"]
                tr = cell.attn_softmax_total_s / cell.attn_matmul_total_s
                fr = cell.flops_attn_softmax / cell.flops_attn_matmul
                lines[i] = (
                    f"| medium | {ctx} | {_format_seconds(cell.attn_matmul_total_s)} | "
                    f"{_format_seconds(cell.attn_softmax_total_s)} | {tr:.3f} | "
                    f"{cell.flops_attn_matmul:.2e} | {cell.flops_attn_softmax:.2e} | {fr:.4f} |"
                )

    lines += _symbol_glossary_lines()

    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    # Drop a previous (c)(d)(e) append if re-running.
    marker = "\n---\n\n## (c) Non-matmul kernels"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + "\n"
    gloss_marker = "\n## 符号表"
    if gloss_marker in existing and marker not in existing:
        existing = existing.split(gloss_marker)[0].rstrip() + "\n"
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] appended (c)(d)(e) → {path.relative_to(REPO_ROOT)}", flush=True)


def _symbol_glossary_lines() -> list[str]:
    """Appendix: symbols used in this report."""
    return [
        "## 符号表",
        "",
        "下文按报告里出现过的符号整理（人话版）。`medium` 预设下部分量有具体数值，便于对照。",
        "",
        "| 符号 | 含义 |",
        "|---|---|",
        "| **B** | batch size，一批里有几条序列。本报告固定 **B = 4**。 |",
        "| **S** | sequence / context length，一条序列的 token 数（上下文长度）。本报告取 256 / 512 / 1024。 |",
        "| **L** | Transformer 层数（`num_layers`）。`medium` 为 **L = 24**。 |",
        "| **H** | attention head 数（`num_heads`）。`medium` 为 **H = 16**。 |",
        "| **d_model** | 模型隐藏维度（残差流宽度）。`medium` 为 **1024**。 |",
        "| **d_k** | 每个 head 的 key/query 维度，通常 `d_k = d_model / H`。`medium` 为 **64**。 |",
        "| **d_v** | 每个 head 的 value 维度；本实现里 **d_v = d_k**。 |",
        "| **Q** | Query 张量（查询），形状大致 `(…, H, S, d_k)`。 |",
        "| **K** | Key 张量（键），形状与 Q 对齐。 |",
        "| **V** | Value 张量（值），形状大致 `(…, H, S, d_v)`。 |",
        "| **A** / 注意力权重 | `softmax(QKᵀ / √d_k)` 得到的概率矩阵，形状 `(…, H, S, S)`，再与 V 相乘。 |",
        "| **QKᵀ** | Query 与 Key 的矩阵乘，得到注意力分数（logits），再除以 `√d_k`。 |",
        "| **AV** | 注意力权重与 Value 的矩阵乘，得到每个 head 的输出。 |",
        "| **√d_k** / `1/√d_k` | 缩放因子，防止点积过大导致 softmax 饱和。 |",
        "| **S²** | 序列长度的平方；注意力分数矩阵每个 head 是 `S×S`，故时间和显存常随 `S²` 涨。 |",
        "| **FLOPs** | floating-point operations，浮点运算次数（算力账本，不是墙上时间）。 |",
        "| **GFLOP** | 10⁹ 次浮点运算；图里右轴把 10 次 forward 的 FLOPs 换成 GFLOP 便于画柱。 |",
        "| **GEMM / SGEMM** | 通用矩阵乘 / FP32 通用矩阵乘（Single-precision GEMM）。 |",
        "| **NVTX** | NVIDIA Tools Extension：在代码里打时间范围标签，供 nsys 聚合。 |",
        "| **attn_matmul** | 本报告给 `QKᵀ` + `AV` 打的 NVTX 名。 |",
        "| **attn_softmax** | 本报告给 attention 内 softmax 打的 NVTX 名。 |",
        "| **train_step** | 完整训练一步的 NVTX：`zero_grad` → forward → loss → backward → AdamW。 |",
        "| **forward** | 仅模型前向（含 `cuda.synchronize`）的 NVTX。 |",
        "| **HBM** | GPU 高带宽显存；带宽受限时，算得少也可能很慢。 |",
        "| **tile / 128×128** | GEMM 把大矩阵切成小块计算；名字里的数字是切块大小。 |",
        "| **tn / nn** | cuBLAS 布局：`t`=转置，`n`=不转置；描述左右矩阵要不要转置。 |",
        "",
        "常用关系（本报告 `medium`）：`d_k = d_model / H`；attention matmul FLOPs 每层约 "
        "`4·B·H·S²·d_k`；softmax 约 `5·B·H·S²`；故 soft/mm ≈ `5/(4·d_k)`。",
        "",
    ]


def run_suite_cde() -> None:
    CDE_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_dir = CDE_ARTIFACTS / f"suite_{stamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    cells: list[CdeCellResult] = []
    total = len(CDE_SIZES) * len(CDE_CONTEXTS)
    idx = 0
    for size in CDE_SIZES:
        for ctx in CDE_CONTEXTS:
            idx += 1
            print(f"\n======== CDE {idx}/{total}: {size} ctx={ctx} ========", flush=True)
            cell = run_cde_cell(size, ctx, suite_dir / f"{size}_ctx{ctx}")
            cells.append(cell)
            print(f"[cde] done {size} ctx={ctx} oom={cell.oom} notes={cell.notes!r}", flush=True)

    part_c = plot_part_c(FIGURE_C)
    plot_part_d(cells, FIGURE_D)
    plot_part_e(cells, FIGURE_E)
    append_cde_report(cells, part_c)

    manifest = {
        "suite_dir": str(suite_dir.relative_to(REPO_ROOT)),
        "cells": [c.to_dict() for c in cells],
        "part_c": part_c,
        "figures": {
            "c": f"reports/figures/{FIGURE_C.name}",
            "d": f"reports/figures/{FIGURE_D.name}",
            "e": f"reports/figures/{FIGURE_E.name}",
        },
        "report": "reports/nsys-profile.md",
    }
    CDE_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[cde] done → reports/nsys-profile.md + nsys_c/d/e_*.png", flush=True)


def replot_cde(manifest_path: Path | None = None) -> None:
    path = manifest_path or CDE_MANIFEST
    man = json.loads(path.read_text(encoding="utf-8"))
    cells = [CdeCellResult.from_dict(c) for c in man["cells"]]
    part_c = plot_part_c(FIGURE_C)
    plot_part_d(cells, FIGURE_D)
    plot_part_e(cells, FIGURE_E)
    append_cde_report(cells, part_c)
    man["part_c"] = part_c
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="nsys_profile parts (c)(d)(e)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_w = sub.add_parser("workload", help="Full train step + attn NVTX (under nsys)")
    sp_w.add_argument("--model-size", required=True, choices=list(MODEL_SIZE_PRESETS))
    sp_w.add_argument("--context-length", type=int, required=True)
    sp_w.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    sp_w.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    sp_w.add_argument("--seed", type=int, default=DEFAULT_SEED)

    sub.add_parser("suite", help="Run medium×{256,512,1024} train profiles + plots + report append")
    sub.add_parser("replot", help="Regenerate (c)(d)(e) plots/report from cde manifest")

    args = p.parse_args(argv)
    if args.cmd == "workload":
        cfg = _cfg(args.model_size, args.context_length, args.warmup, args.steps)
        cfg.seed = args.seed
        run_train_workload(cfg)
    elif args.cmd == "suite":
        run_suite_cde()
    elif args.cmd == "replot":
        replot_cde()
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
