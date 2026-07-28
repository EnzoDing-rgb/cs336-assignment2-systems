"""Profile peak memory for checkpointed xl training steps + sweep CLI."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.e2e_timing.e2e import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_VOCAB_SIZE,
    BenchmarkConfig,
    build_model,
    make_batch,
)
from cs336_systems.gradient_checkpointing.forward import forward_lm_with_checkpoint
from cs336_systems.gradient_checkpointing.plots import make_figures

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "gradient_checkpointing"
REPORT_PATH = REPO_ROOT / "reports" / "gradient-checkpointing.md"

DEFAULT_SEGMENT_SIZES: tuple[int | None, ...] = (None, 1, 2, 4, 8, 16, 32)
APPENDIX_CONTEXT = 512
PRIMARY_CONTEXT = 2048


@dataclass(frozen=True)
class CheckpointProfileConfig:
    model_size: str = "xl"
    batch_size: int = DEFAULT_BATCH_SIZE
    context_length: int = PRIMARY_CONTEXT
    segment_size: int | None = None
    vocab_size: int = DEFAULT_VOCAB_SIZE
    warmup: int = 1
    seed: int = DEFAULT_SEED


@dataclass
class ProfileResult:
    segment_size: int | None
    peak_allocated_bytes: int | None
    peak_allocated_gib: float | None
    oom: bool
    error: str | None
    context_length: int
    batch_size: int
    model_size: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _warmup_step(
    model: torch.nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    segment_size: int | None,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    logits = forward_lm_with_checkpoint(model, x, segment_size)
    loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    _sync(x.device)


def profile_train_step(cfg: CheckpointProfileConfig, device: torch.device) -> ProfileResult:
    """One full training step; returns peak ``max_memory_allocated`` for this config."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    _cleanup(device)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    bcfg = BenchmarkConfig(
        model_size=cfg.model_size,
        mode="train",
        vocab_size=cfg.vocab_size,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        seed=cfg.seed,
        mixed_precision="off",
        do_optimizer=True,
    )

    try:
        model = build_model(bcfg, device)
        model.train()
        optimizer = AdamW(model.parameters())
        x, y = make_batch(bcfg, device)

        for _ in range(cfg.warmup):
            _warmup_step(model, optimizer, x, y, cfg.segment_size)

        _cleanup(device)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_lm_with_checkpoint(model, x, cfg.segment_size)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        _sync(device)

        peak = int(torch.cuda.max_memory_allocated())
        return ProfileResult(
            segment_size=cfg.segment_size,
            peak_allocated_bytes=peak,
            peak_allocated_gib=peak / (1024**3),
            oom=False,
            error=None,
            context_length=cfg.context_length,
            batch_size=cfg.batch_size,
            model_size=cfg.model_size,
        )
    except torch.cuda.OutOfMemoryError as err:
        return ProfileResult(
            segment_size=cfg.segment_size,
            peak_allocated_bytes=None,
            peak_allocated_gib=None,
            oom=True,
            error=str(err),
            context_length=cfg.context_length,
            batch_size=cfg.batch_size,
            model_size=cfg.model_size,
        )
    finally:
        try:
            del model, optimizer, x, y, logits, loss  # type: ignore[possibly-unbound]
        except NameError:
            pass
        _cleanup(device)


def run_sweep(
    segment_sizes: tuple[int | None, ...] = DEFAULT_SEGMENT_SIZES,
    *,
    base_cfg: CheckpointProfileConfig | None = None,
    device: torch.device | None = None,
) -> list[ProfileResult]:
    """Run profile_train_step for each segment_size (None = baseline)."""
    if device is None:
        device = torch.device("cuda")
    if base_cfg is None:
        base_cfg = CheckpointProfileConfig()

    results: list[ProfileResult] = []
    for seg in segment_sizes:
        cfg = CheckpointProfileConfig(
            model_size=base_cfg.model_size,
            batch_size=base_cfg.batch_size,
            context_length=base_cfg.context_length,
            segment_size=seg,
            vocab_size=base_cfg.vocab_size,
            warmup=base_cfg.warmup,
            seed=base_cfg.seed,
        )
        print(f"  segment_size={seg!r} …", flush=True)
        result = profile_train_step(cfg, device)
        if result.oom:
            print("    OOM", flush=True)
        else:
            print(f"    peak {result.peak_allocated_gib:.3f} GiB", flush=True)
        results.append(result)
    return results


def _pick_best_segment(results: list[ProfileResult]) -> int | None:
    ok = [r for r in results if r.segment_size is not None and not r.oom and r.peak_allocated_bytes]
    if not ok:
        return None
    return min(ok, key=lambda r: r.peak_allocated_bytes).segment_size  # type: ignore[arg-type]


def main() -> None:
    p = argparse.ArgumentParser(description="Gradient checkpointing memory sweep")
    p.add_argument("--skip-run", action="store_true", help="Rebuild figures/report from peaks.json")
    p.add_argument("--no-appendix", action="store_true", help="Skip S=512 appendix sweep")
    args = p.parse_args()

    peaks_path = ARTIFACTS_ROOT / "peaks.json"

    if args.skip_run:
        manifest = json.loads(peaks_path.read_text(encoding="utf-8"))
        primary = [ProfileResult(**r) for r in manifest["primary"]]
        appendix_raw = manifest.get("appendix")
        appendix = [ProfileResult(**r) for r in appendix_raw] if appendix_raw else None
    else:
        ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda")

        print("── Primary sweep S=2048 ──", flush=True)
        primary = run_sweep(base_cfg=CheckpointProfileConfig(context_length=PRIMARY_CONTEXT), device=device)

        appendix: list[ProfileResult] | None = None
        if not args.no_appendix:
            print("── Appendix sweep S=512 ──", flush=True)
            appendix = run_sweep(
                base_cfg=CheckpointProfileConfig(context_length=APPENDIX_CONTEXT),
                device=device,
            )

        peaks_path.write_text(
            json.dumps(
                {
                    "primary": [r.to_dict() for r in primary],
                    "appendix": [r.to_dict() for r in appendix] if appendix else None,
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote {peaks_path}", flush=True)

    best_k = _pick_best_segment(primary)
    figs = make_figures(
        [r.to_dict() for r in primary],
        results_appendix=[r.to_dict() for r in appendix] if appendix else None,
        best_segment=best_k,
    )
    print("Figures:", ", ".join(str(p) for p in figs.values()), flush=True)
    print(
        f"Report (hand-maintained): {REPORT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
