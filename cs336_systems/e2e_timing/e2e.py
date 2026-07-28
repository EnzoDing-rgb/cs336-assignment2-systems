"""End-to-end wall-clock timing for Assignment 2 `benchmarking_script`.

Layout:
  cs336_systems/e2e_timing/e2e.py      — single run + suite + plots + report
  cs336_systems/e2e_timing/__main__.py — suite entry

Run one experiment (part a):
  uv run python -m cs336_systems.e2e_timing.e2e --model-size small --mode timed_train

Run full suite (parts b+c):
  uv run python -m cs336_systems.e2e_timing
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SIZE_PRESETS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}
MODEL_SIZE_NAMES: tuple[str, ...] = tuple(MODEL_SIZE_PRESETS.keys())
MODES: tuple[str, ...] = ("forward", "forward_backward", "train", "timed_train")
TIMED_TRAIN_SEGMENTS: tuple[str, ...] = ("forward", "loss", "backward", "optimizer")

DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_BATCH_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_WARMUP = 5
DEFAULT_STEPS = 10
DEFAULT_SEED = 0

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "e2e_benchmark"
REPORT_PATH = REPO_ROOT / "reports" / "end2end-benchmark.md"
FIGURE_B = REPO_ROOT / "reports" / "figures" / "e2e_benchmark_timed_train.png"
FIGURE_STD = REPO_ROOT / "reports" / "figures" / "e2e_benchmark_segment_std.png"
FIGURE_C = REPO_ROOT / "reports" / "figures" / "e2e_benchmark_warmup_ablation.png"
SUITE_MANIFEST = ARTIFACTS_ROOT / "suite_manifest.json"

PART_B_WARMUP = 5
PART_C_WARMUPS = (0, 1, 2)
SUITE_MODE = "timed_train"
SUITE_STEPS = 10
SUITE_SEED = 0

Mode = Literal["forward", "forward_backward", "train", "timed_train"]
MixedPrecision = Literal["off", "bf16"]


def _amp_context(mixed_precision: MixedPrecision):
    """BF16 autocast or no-op. No GradScaler (BF16 does not need it)."""
    if mixed_precision == "off":
        return nullcontext()
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unknown mixed_precision={mixed_precision!r}")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    model_size: str
    mode: Mode
    vocab_size: int = DEFAULT_VOCAB_SIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    context_length: int = DEFAULT_CONTEXT_LENGTH
    warmup: int = DEFAULT_WARMUP
    steps: int = DEFAULT_STEPS
    seed: int = DEFAULT_SEED
    device: str = "cuda"
    d_model: int | None = None
    d_ff: int | None = None
    num_layers: int | None = None
    num_heads: int | None = None
    mixed_precision: MixedPrecision = "off"
    # When False, timed_train still measures forward/loss/backward but skips AdamW.step().
    do_optimizer: bool = True
    use_compile: bool = False

    def resolved_model_hparams(self) -> dict[str, int]:
        if self.model_size not in MODEL_SIZE_PRESETS:
            raise ValueError(f"Unknown model_size={self.model_size!r}. Choose from {list(MODEL_SIZE_PRESETS)}.")
        base = dict(MODEL_SIZE_PRESETS[self.model_size])
        for key, value in {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }.items():
            if value is not None:
                base[key] = value
        return base


@dataclass
class SegmentStats:
    mean_s: float
    std_s: float
    times_s: list[float]


@dataclass
class BenchmarkResult:
    config: dict[str, Any]
    model_hparams: dict[str, int]
    run_id: str
    artifact_dir: str
    wall_time_s: float
    step_times_s: list[float]
    step_mean_s: float
    step_std_s: float
    segments: dict[str, SegmentStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        for stream in self.streams:
            isatty = getattr(stream, "isatty", None)
            if callable(isatty) and isatty():
                return True
        return False


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return statistics.mean(xs), statistics.stdev(xs)


def _format_seconds(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    if x >= 1.0:
        return f"{x:.4f} s"
    return f"{x * 1000:.3f} ms"


def _print_banner(lines: list[str]) -> None:
    width = max(len(line) for line in lines)
    bar = "═" * (width + 2)
    print(f"╔{bar}╗")
    for line in lines:
        print(f"║ {line.ljust(width)} ║")
    print(f"╚{bar}╝")


def _print_kv_section(title: str, rows: list[tuple[str, str]]) -> None:
    key_w = max(len(k) for k, _ in rows)
    print()
    print(f"── {title} ──")
    for key, value in rows:
        print(f"  {key.ljust(key_w)}  {value}")


# ---------------------------------------------------------------------------
# Single-run benchmark
# ---------------------------------------------------------------------------


def build_model(cfg: BenchmarkConfig, device: torch.device) -> BasicsTransformerLM:
    hparams = cfg.resolved_model_hparams()
    model = BasicsTransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=hparams["d_model"],
        num_layers=hparams["num_layers"],
        num_heads=hparams["num_heads"],
        d_ff=hparams["d_ff"],
    )
    return model.to(device)


def make_batch(cfg: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.context_length), device=device)
    return x, y


def _run_forward(
    model: BasicsTransformerLM,
    x: torch.Tensor,
    *,
    mixed_precision: MixedPrecision = "off",
) -> torch.Tensor:
    with _amp_context(mixed_precision):
        return model(x)


def _run_forward_backward(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mixed_precision: MixedPrecision = "off",
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(mixed_precision):
        logits = model(x)
        loss = cross_entropy(logits, y)
    loss.backward()


def _run_train(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mixed_precision: MixedPrecision = "off",
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with _amp_context(mixed_precision):
        logits = model(x)
        loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()


def _run_timed_train_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    mixed_precision: MixedPrecision = "off",
    do_optimizer: bool = True,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    amp = _amp_context(mixed_precision)

    _sync(device)
    t0 = timeit.default_timer()
    with amp:
        logits = model(x)
    _sync(device)
    forward_s = timeit.default_timer() - t0

    _sync(device)
    t0 = timeit.default_timer()
    with amp:
        loss = cross_entropy(logits, y)
    _sync(device)
    loss_s = timeit.default_timer() - t0

    _sync(device)
    t0 = timeit.default_timer()
    loss.backward()
    _sync(device)
    backward_s = timeit.default_timer() - t0

    if do_optimizer:
        _sync(device)
        t0 = timeit.default_timer()
        optimizer.step()
        _sync(device)
        optimizer_s = timeit.default_timer() - t0
    else:
        optimizer_s = 0.0

    return {"forward": forward_s, "loss": loss_s, "backward": backward_s, "optimizer": optimizer_s}


def run_benchmark(cfg: BenchmarkConfig, artifacts_root: Path | None = None) -> BenchmarkResult:
    if cfg.mode not in MODES:
        raise ValueError(f"Unknown mode={cfg.mode!r}. Choose from {list(MODES)}.")
    if cfg.device != "cuda":
        raise ValueError("This benchmark requires --device cuda.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.cuda.empty_cache()

    model_hparams = cfg.resolved_model_hparams()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{timestamp}_{cfg.model_size}_{cfg.mode}_wu{cfg.warmup}"
        f"_mp{cfg.mixed_precision}_opt{int(cfg.do_optimizer)}"
    )
    root = artifacts_root if artifacts_root is not None else ARTIFACTS_ROOT
    artifact_dir = root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "run.log"

    model = None
    optimizer = None
    wall_start = timeit.default_timer()
    with log_path.open("w", encoding="utf-8") as log_file:
        tee = _Tee(sys.stdout, log_file)
        old_stdout = sys.stdout
        sys.stdout = tee  # type: ignore[assignment]
        try:
            _print_banner(
                [
                    "E2E Benchmark",
                    f"model_size={cfg.model_size}  mode={cfg.mode}",
                    f"warmup={cfg.warmup}  steps={cfg.steps}  seed={cfg.seed}",
                ]
            )
            _print_kv_section(
                "Config",
                [
                    ("vocab_size", str(cfg.vocab_size)),
                    ("batch_size", str(cfg.batch_size)),
                    ("context_length", str(cfg.context_length)),
                    ("device", str(device)),
                    ("d_model", str(model_hparams["d_model"])),
                    ("d_ff", str(model_hparams["d_ff"])),
                    ("num_layers", str(model_hparams["num_layers"])),
                    ("num_heads", str(model_hparams["num_heads"])),
                    ("mixed_precision", cfg.mixed_precision),
                    ("do_optimizer", str(cfg.do_optimizer)),
                    ("use_compile", str(cfg.use_compile)),
                    ("artifact_dir", str(artifact_dir)),
                ],
            )

            print("\n── Build ──")
            print("  initializing model …")
            model = build_model(cfg, device)
            if cfg.use_compile:
                print("  torch.compile(model) …")
                model = torch.compile(model)
            model.train()
            optimizer = AdamW(model.parameters())
            x, y = make_batch(cfg, device)
            print("  model ready; batch allocated and reused for all steps.")

            def one_whole_step() -> None:
                if cfg.mode == "forward":
                    _run_forward(model, x, mixed_precision=cfg.mixed_precision)
                elif cfg.mode == "forward_backward":
                    _run_forward_backward(
                        model, optimizer, x, y, mixed_precision=cfg.mixed_precision
                    )
                elif cfg.mode == "train":
                    _run_train(model, optimizer, x, y, mixed_precision=cfg.mixed_precision)
                else:
                    raise AssertionError("timed_train uses segmented path")

            print("\n── Warmup ──")
            if cfg.warmup == 0:
                print("  (no warmup)")
            for i in range(cfg.warmup):
                if cfg.mode == "timed_train":
                    _run_timed_train_step(
                        model,
                        optimizer,
                        x,
                        y,
                        device,
                        mixed_precision=cfg.mixed_precision,
                        do_optimizer=cfg.do_optimizer,
                    )
                else:
                    one_whole_step()
                    _sync(device)
                print(f"  warmup {i + 1}/{cfg.warmup} done")

            print("\n── Measure ──")
            step_times: list[float] = []
            segment_times: dict[str, list[float]] = {name: [] for name in TIMED_TRAIN_SEGMENTS}

            for i in range(cfg.steps):
                if cfg.mode == "timed_train":
                    parts = _run_timed_train_step(
                        model,
                        optimizer,
                        x,
                        y,
                        device,
                        mixed_precision=cfg.mixed_precision,
                        do_optimizer=cfg.do_optimizer,
                    )
                    for name in TIMED_TRAIN_SEGMENTS:
                        segment_times[name].append(parts[name])
                    total = sum(parts[name] for name in TIMED_TRAIN_SEGMENTS)
                    step_times.append(total)
                    print(
                        f"  step {i + 1:>2}/{cfg.steps}  "
                        f"fwd={_format_seconds(parts['forward'])}  "
                        f"loss={_format_seconds(parts['loss'])}  "
                        f"bwd={_format_seconds(parts['backward'])}  "
                        f"opt={_format_seconds(parts['optimizer'])}  "
                        f"sum={_format_seconds(total)}"
                    )
                else:
                    _sync(device)
                    t0 = timeit.default_timer()
                    one_whole_step()
                    _sync(device)
                    dt = timeit.default_timer() - t0
                    step_times.append(dt)
                    print(f"  step {i + 1:>2}/{cfg.steps}  {_format_seconds(dt)}")

            step_mean, step_std = _mean_std(step_times)
            segments: dict[str, SegmentStats] = {}
            if cfg.mode == "timed_train":
                for name in TIMED_TRAIN_SEGMENTS:
                    mean_s, std_s = _mean_std(segment_times[name])
                    segments[name] = SegmentStats(mean_s=mean_s, std_s=std_s, times_s=segment_times[name])

            wall_time_s = timeit.default_timer() - wall_start
            summary_rows = [
                ("step mean", _format_seconds(step_mean)),
                ("step std", _format_seconds(step_std)),
                ("wall time", _format_seconds(wall_time_s)),
            ]
            for name, stats in segments.items():
                summary_rows.append((f"{name} mean", _format_seconds(stats.mean_s)))
                summary_rows.append((f"{name} std", _format_seconds(stats.std_s)))
            _print_kv_section("Summary", summary_rows)

            result = BenchmarkResult(
                config={
                    "model_size": cfg.model_size,
                    "mode": cfg.mode,
                    "vocab_size": cfg.vocab_size,
                    "batch_size": cfg.batch_size,
                    "context_length": cfg.context_length,
                    "warmup": cfg.warmup,
                    "steps": cfg.steps,
                    "seed": cfg.seed,
                    "device": cfg.device,
                    "mixed_precision": cfg.mixed_precision,
                    "do_optimizer": cfg.do_optimizer,
                },
                model_hparams=model_hparams,
                run_id=run_id,
                artifact_dir=str(artifact_dir),
                wall_time_s=wall_time_s,
                step_times_s=step_times,
                step_mean_s=step_mean,
                step_std_s=step_std,
                segments=segments,
            )
            summary_path = artifact_dir / "summary.json"
            summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            print(f"\nWrote {summary_path}")
            return result
        finally:
            sys.stdout = old_stdout
            # Drop references so the next suite cell can fit.
            try:
                del model
            except NameError:
                pass
            try:
                del optimizer
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Plots + report
# ---------------------------------------------------------------------------


def plot_timed_train_by_size(results_wu5: list[dict[str, Any]], out_path: Path) -> None:
    ok = [r for r in results_wu5 if r.get("segments") and not r.get("oom")]
    by_size = {r["config"]["model_size"]: r for r in ok}
    sizes = [s for s in MODEL_SIZE_NAMES if s in by_size]
    if not sizes:
        print(f"[plot] skip {out_path.name}: no successful timed_train results", flush=True)
        return
    x = np.arange(len(sizes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, seg in enumerate(TIMED_TRAIN_SEGMENTS):
        means = [by_size[s]["segments"][seg]["mean_s"] for s in sizes]
        ax.bar(x + (i - 1.5) * width, means, width, label=seg)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Model size")
    ax.set_title("timed_train segment times (warmup=5, steps=10)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_segment_std(results_wu5: list[dict[str, Any]], out_path: Path) -> None:
    """Grouped bars: x=model size, series=per-segment standard deviation (seconds)."""
    ok = [r for r in results_wu5 if r.get("segments") and not r.get("oom")]
    by_size = {r["config"]["model_size"]: r for r in ok}
    sizes = [s for s in MODEL_SIZE_NAMES if s in by_size]
    if not sizes:
        print(f"[plot] skip {out_path.name}: no successful timed_train results", flush=True)
        return
    x = np.arange(len(sizes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, seg in enumerate(TIMED_TRAIN_SEGMENTS):
        stds = [by_size[s]["segments"][seg]["std_s"] for s in sizes]
        ax.bar(x + (i - 1.5) * width, stds, width, label=seg)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("Std (s)")
    ax.set_xlabel("Model size")
    ax.set_title("timed_train segment std over 10 steps (warmup=5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_warmup_ablation(results: list[dict[str, Any]], out_path: Path) -> None:
    """Full-step *mean time* vs warmup count (not std; not wall-clock of warmup+measure).

    For each warmup=W: discard W steps, then average the next 10 measured full steps.
    Error bars are std over those 10 measured steps.
    """
    ok = [r for r in results if not r.get("oom") and "step_mean_s" in r]
    warmups = sorted({r["config"]["warmup"] for r in ok})
    sizes = [s for s in MODEL_SIZE_NAMES if any(r["config"]["model_size"] == s for r in ok)]
    if not sizes:
        print(f"[plot] skip {out_path.name}: no successful results", flush=True)
        return
    n = len(sizes)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, size in zip(axes, sizes):
        xs, ys, yerr = [], [], []
        for wu in warmups:
            match = next(
                (r for r in ok if r["config"]["model_size"] == size and r["config"]["warmup"] == wu),
                None,
            )
            if match is None:
                continue
            xs.append(wu)
            ys.append(match["step_mean_s"])
            yerr.append(match.get("step_std_s", 0.0))
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, linewidth=1.5)
        ax.set_title(size)
        ax.set_xlabel("# warmup steps (discarded)")
        ax.set_xticks(warmups)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Mean time per measured step (s)\n(= forward+loss+backward+optimizer)")
    fig.suptitle(
        "Warmup ablation (TIME, not std): after W warmup steps, mean of next 10 full steps",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _fmt(x: float) -> str:
    return _format_seconds(x)


def _row_for_size(result: dict[str, Any]) -> str:
    cells = [result["config"]["model_size"]]
    for seg in TIMED_TRAIN_SEGMENTS:
        stats = result["segments"][seg]
        cells.append(f"{_fmt(stats['mean_s'])} ± {_fmt(stats['std_s'])}")
    cells.append(f"{_fmt(result['step_mean_s'])} ± {_fmt(result['step_std_s'])}")
    return "| " + " | ".join(cells) + " |"


def write_report(
    path: Path,
    part_b_results: list[dict[str, Any]],
    part_c_results: list[dict[str, Any]],
    figure_b: str,
    figure_std: str,
    figure_c: str,
) -> None:
    ok_b = [r for r in part_b_results if r.get("segments") and not r.get("oom")]
    by_size_b = {r["config"]["model_size"]: r for r in ok_b}
    if not ok_b:
        raise RuntimeError("No successful part-(b) results to write a report from.")
    focus = by_size_b.get("medium") or ok_b[0]
    fwd = focus["segments"]["forward"]
    bwd = focus["segments"]["backward"]

    ratios = []
    for r in ok_b:
        for seg in TIMED_TRAIN_SEGMENTS:
            mean = r["segments"][seg]["mean_s"]
            std = r["segments"][seg]["std_s"]
            if mean > 0:
                ratios.append(std / mean)
    max_cv = max(ratios) if ratios else float("nan")
    variability = "small" if max_cv < 0.05 else "noticeable"

    oom_sizes = sorted(
        {
            r["config"]["model_size"]
            for r in part_b_results + part_c_results
            if r.get("oom")
        }
    )
    oom_note = (
        f"**Note:** model size(s) {', '.join(oom_sizes)} hit CUDA OOM on this 80GB GPU "
        f"(fp32 + AdamW + batch=4 + context=512) and are omitted from timing plots/tables."
        if oom_sizes
        else ""
    )

    def step_mean(size: str, wu: int) -> float | None:
        for r in part_c_results:
            if r.get("oom"):
                continue
            if r["config"]["model_size"] == size and r["config"]["warmup"] == wu:
                return r["step_mean_s"]
        return None

    narr_size = (
        "medium"
        if any(r["config"]["model_size"] == "medium" and not r.get("oom") for r in part_c_results)
        else ok_b[0]["config"]["model_size"]
    )
    m0, m1, m2, m5 = (step_mean(narr_size, w) for w in (0, 1, 2, 5))

    body_b = [
        "| size | forward | loss | backward | optimizer | full step |",
        "|---|---|---|---|---|---|",
    ]
    for size in MODEL_SIZE_NAMES:
        if size in by_size_b:
            body_b.append(_row_for_size(by_size_b[size]))
        elif any(r.get("oom") and r["config"]["model_size"] == size for r in part_b_results):
            body_b.append(f"| {size} | OOM | OOM | OOM | OOM | OOM |")

    warmups = [0, 1, 2, 5]
    c_rows = [
        "| size | " + " | ".join(f"warmup={w}" for w in warmups) + " |",
        "|---|" + "|".join(["---"] * len(warmups)) + "|",
    ]
    for size in MODEL_SIZE_NAMES:
        cells = [size]
        for wu in warmups:
            match = next(
                (
                    r
                    for r in part_c_results
                    if r["config"]["model_size"] == size and r["config"]["warmup"] == wu
                ),
                None,
            )
            if match is None:
                cells.append("—")
            elif match.get("oom"):
                cells.append("OOM")
            else:
                cells.append(f"{_fmt(match['step_mean_s'])} ± {_fmt(match['step_std_s'])}")
        c_rows.append("| " + " | ".join(cells) + " |")

    md = f"""# End-to-End Benchmark Report

Assignment 2 `benchmarking_script` parts (b) and (c).

**Setup:** `BasicsTransformerLM`, vocab=10000, batch=4, context=512, AdamW, CUDA, `timed_train` segmented timing (forward / loss / backward / optimizer), `timeit.default_timer` + `torch.cuda.synchronize` per segment. Measurement steps=10.

**How to reproduce:**

```bash
uv run --no-sync python -m cs336_systems.e2e_timing
```

{oom_note}

## (b) Timings with warmup=5

![timed_train mean by size]({figure_b})

![timed_train std by size]({figure_std})

{chr(10).join(body_b)}

表中每个格子是 **mean ± std**（10 次测量）。方差 = std²，不另列表。

**Answer (b):** On the `{focus["config"]["model_size"]}` model, a forward pass takes {_fmt(fwd["mean_s"])} (std {_fmt(fwd["std_s"])}) and a backward pass takes {_fmt(bwd["mean_s"])} (std {_fmt(bwd["std_s"])}). Across sizes and segments the relative std (std/mean) is {variability} (max observed ≈ {max_cv:.3f}), so run-to-run variability after warmup is generally low.

## (c) Warmup ablation

Warmup ∈ {{0,1,2,5}}; warmup=5 reused from (b).

**What the y-axis is:** **time** (seconds), not std. For each warmup=`W` we **discard** the first `W` steps (untimed), then time the next 10 full steps and plot their **mean**. Warmup duration itself is **not** included in the plotted number. Error bars = std over those 10 measured steps.

![warmup ablation]({figure_c})

{chr(10).join(c_rows)}

**Answer (c):** Without warmup (warmup=0), measured full-step time on `{narr_size}` is {_fmt(m0) if m0 is not None else "n/a"} versus {_fmt(m5) if m5 is not None else "n/a"} at warmup=5, typically higher and/or noisier because the first steps pay one-time GPU costs (context init, kernel selection/caching, allocator warmup). With only 1–2 warmup steps (here {_fmt(m1) if m1 is not None else "n/a"} / {_fmt(m2) if m2 is not None else "n/a"}), results can still differ from warmup=5 because those transient effects may not have fully settled yet. Small dips at warmup=2 vs warmup=5 (when present) are within timing noise (see error bars / ±std), not evidence that 2 warmups is systematically faster.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")


# ---------------------------------------------------------------------------
# Suite (parts b + c)
# ---------------------------------------------------------------------------


def _load_latest_summary(size: str, mode: str, warmup: int) -> dict[str, Any] | None:
    matches = sorted(ARTIFACTS_ROOT.glob(f"*_{size}_{mode}_wu{warmup}/summary.json"))
    if not matches:
        return None
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def _oom_record(cfg: BenchmarkConfig, err: BaseException) -> dict[str, Any]:
    return {
        "config": {
            "model_size": cfg.model_size,
            "mode": cfg.mode,
            "vocab_size": cfg.vocab_size,
            "batch_size": cfg.batch_size,
            "context_length": cfg.context_length,
            "warmup": cfg.warmup,
            "steps": cfg.steps,
            "seed": cfg.seed,
            "device": cfg.device,
        },
        "model_hparams": cfg.resolved_model_hparams(),
        "oom": True,
        "error": str(err),
        "run_id": f"oom_{cfg.model_size}_{cfg.mode}_wu{cfg.warmup}",
    }


def _run_cell(cfg: BenchmarkConfig, *, reuse: bool) -> dict[str, Any]:
    if reuse:
        existing = _load_latest_summary(cfg.model_size, cfg.mode, cfg.warmup)
        if existing is not None and existing.get("oom"):
            print(
                f"[suite {time.strftime('%H:%M:%S')}] reuse OOM record "
                f"{existing.get('run_id')} ({cfg.model_size} wu={cfg.warmup})",
                flush=True,
            )
            return existing
        if existing is not None and existing.get("segments") and not existing.get("oom"):
            print(
                f"[suite {time.strftime('%H:%M:%S')}] reuse existing "
                f"{existing.get('run_id')} ({cfg.model_size} wu={cfg.warmup})",
                flush=True,
            )
            return existing
    try:
        return run_benchmark(cfg, artifacts_root=ARTIFACTS_ROOT).to_dict()
    except torch.cuda.OutOfMemoryError as err:
        print(f"[suite {time.strftime('%H:%M:%S')}] OOM on {cfg.model_size} wu={cfg.warmup}: {err}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        rec = _oom_record(cfg, err)
        oom_dir = ARTIFACTS_ROOT / rec["run_id"]
        oom_dir.mkdir(parents=True, exist_ok=True)
        (oom_dir / "summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec


def run_suite() -> None:
    print(f"[suite {time.strftime('%H:%M:%S')}] run_suite() begin", flush=True)
    print(f"[suite {time.strftime('%H:%M:%S')}] artifacts -> {ARTIFACTS_ROOT}", flush=True)
    print(f"[suite {time.strftime('%H:%M:%S')}] report    -> reports/end2end-benchmark.md", flush=True)
    ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("SUITE PART (b): timed_train × all sizes × warmup=5 (reuse if present)", flush=True)
    print("=" * 72, flush=True)
    part_b: list[dict[str, Any]] = []
    for idx, size in enumerate(MODEL_SIZE_NAMES, start=1):
        print(
            f"[suite {time.strftime('%H:%M:%S')}] (b) {idx}/{len(MODEL_SIZE_NAMES)} "
            f"model_size={size} warmup={PART_B_WARMUP}",
            flush=True,
        )
        cfg = BenchmarkConfig(model_size=size, mode=SUITE_MODE, warmup=PART_B_WARMUP, steps=SUITE_STEPS, seed=SUITE_SEED)
        result = _run_cell(cfg, reuse=True)
        part_b.append(result)
        status = "OOM" if result.get("oom") else "ok"
        print(f"[suite {time.strftime('%H:%M:%S')}] (b) finished {size} ({status})", flush=True)

    print("=" * 72, flush=True)
    print("SUITE PART (c): timed_train × all sizes × warmup in {0,1,2}", flush=True)
    print("warmup=5 is reused from part (b); not re-run.", flush=True)
    print("=" * 72, flush=True)
    part_c_extra: list[dict[str, Any]] = []
    total_c = len(MODEL_SIZE_NAMES) * len(PART_C_WARMUPS)
    done_c = 0
    for size in MODEL_SIZE_NAMES:
        # If part (b) already OOMed this size, don't waste time on (c).
        b_hit = next((r for r in part_b if r["config"]["model_size"] == size), None)
        for warmup in PART_C_WARMUPS:
            done_c += 1
            print(
                f"[suite {time.strftime('%H:%M:%S')}] (c) {done_c}/{total_c} "
                f"model_size={size} warmup={warmup}",
                flush=True,
            )
            cfg = BenchmarkConfig(model_size=size, mode=SUITE_MODE, warmup=warmup, steps=SUITE_STEPS, seed=SUITE_SEED)
            if b_hit is not None and b_hit.get("oom"):
                print(f"[suite {time.strftime('%H:%M:%S')}] skip {size} wu={warmup}: part (b) already OOM", flush=True)
                part_c_extra.append(_oom_record(cfg, RuntimeError("skipped: part (b) OOM")))
                continue
            result = _run_cell(cfg, reuse=True)
            part_c_extra.append(result)
            status = "OOM" if result.get("oom") else "ok"
            print(f"[suite {time.strftime('%H:%M:%S')}] (c) finished size={size} warmup={warmup} ({status})", flush=True)

    part_c = part_c_extra + part_b
    plot_timed_train_by_size(part_b, FIGURE_B)
    plot_segment_std(part_b, FIGURE_STD)
    plot_warmup_ablation(part_c, FIGURE_C)
    write_report(
        REPORT_PATH,
        part_b_results=part_b,
        part_c_results=part_c,
        figure_b="figures/e2e_benchmark_timed_train.png",
        figure_std="figures/e2e_benchmark_segment_std.png",
        figure_c="figures/e2e_benchmark_warmup_ablation.png",
    )
    SUITE_MANIFEST.write_text(
        json.dumps(
            {
                "part_b_run_ids": [r.get("run_id") for r in part_b],
                "part_c_extra_run_ids": [r.get("run_id") for r in part_c_extra],
                "oom_sizes": sorted({r["config"]["model_size"] for r in part_b if r.get("oom")}),
                "report": "reports/end2end-benchmark.md",
                "figure_b": "reports/figures/e2e_benchmark_timed_train.png",
                "figure_std": "reports/figures/e2e_benchmark_segment_std.png",
                "figure_c": "reports/figures/e2e_benchmark_warmup_ablation.png",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=" * 72)
    print("SUITE DONE")
    print("  report  : reports/end2end-benchmark.md")
    print("  figure  : reports/figures/e2e_benchmark_timed_train.png")
    print("  figure  : reports/figures/e2e_benchmark_segment_std.png")
    print("  figure  : reports/figures/e2e_benchmark_warmup_ablation.png")
    print(f"  manifest: {SUITE_MANIFEST}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI: single run
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end Transformer benchmark (single run, one mode).")
    p.add_argument("--model-size", required=True, choices=MODEL_SIZE_NAMES)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="cuda", choices=["cuda"])
    p.add_argument("--d-model", type=int, default=None)
    p.add_argument("--d-ff", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument(
        "--mixed-precision",
        default="off",
        choices=["off", "bf16"],
        help="off=FP32; bf16=torch.autocast bfloat16 (no GradScaler)",
    )
    p.add_argument(
        "--no-optimizer",
        action="store_true",
        help="For timed_train: skip AdamW.step() (still time forward/loss/backward)",
    )
    return p.parse_args()


def main() -> None:
    print(f"[e2e {time.strftime('%H:%M:%S')}] single-run CLI starting", flush=True)
    args = parse_args()
    print(
        f"[e2e {time.strftime('%H:%M:%S')}] parsed args: "
        f"size={args.model_size} mode={args.mode} warmup={args.warmup} steps={args.steps} "
        f"mp={args.mixed_precision}",
        flush=True,
    )
    cfg = BenchmarkConfig(
        model_size=args.model_size,
        mode=args.mode,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        warmup=args.warmup,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mixed_precision=args.mixed_precision,
        do_optimizer=not args.no_optimizer,
    )
    run_benchmark(cfg, artifacts_root=ARTIFACTS_ROOT)


if __name__ == "__main__":
    main()
