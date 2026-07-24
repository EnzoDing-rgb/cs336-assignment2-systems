"""Single-cell CUDA memory profile: staged peaks + mandatory snapshot pickle."""

from __future__ import annotations

import gc
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import torch
from cs336_basics.optimizer import AdamW

from cs336_systems.e2e_timing.e2e import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SEED,
    DEFAULT_VOCAB_SIZE,
    MODEL_SIZE_PRESETS,
    MixedPrecision,
    _amp_context,
    build_model,
    make_batch,
)
from cs336_basics.nn_utils import cross_entropy

Mode = Literal["forward", "train"]


@dataclass(frozen=True)
class MemoryCellConfig:
    model_size: str = "xl"
    context_length: int = 512
    batch_size: int = DEFAULT_BATCH_SIZE
    vocab_size: int = DEFAULT_VOCAB_SIZE
    mode: Mode = "train"
    mixed_precision: MixedPrecision = "off"
    warmup: int = 2
    seed: int = DEFAULT_SEED

    @property
    def run_id(self) -> str:
        mp = self.mixed_precision
        return f"{self.model_size}_ctx{self.context_length}_{self.mode}_mp{mp}"


def _sync() -> None:
    torch.cuda.synchronize()


def _now_us() -> int:
    """Wall clock in microseconds — matches PyTorch snapshot ``time_us`` (Unix µs)."""
    return time.time_ns() // 1000


def _mem_snapshot() -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


@contextmanager
def _record_history(max_entries: int = 1_000_000) -> Iterator[None]:
    # stacks="python" keeps assignment-readable frames for memory_viz / (e).
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=max_entries,
    )
    try:
        yield
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def _warmup(
    model: torch.nn.Module,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    mode: Mode,
    mixed_precision: MixedPrecision,
    n: int,
) -> None:
    amp = _amp_context(mixed_precision)
    for _ in range(n):
        if mode == "forward":
            with amp:
                _ = model(x)
        else:
            optimizer.zero_grad(set_to_none=True)
            with amp:
                logits = model(x)
                loss = cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        _sync()


def profile_cell(cfg: MemoryCellConfig, out_dir: Path) -> dict[str, Any]:
    """Run one memory-profile cell; always writes a pickle + stage metrics JSON fields."""
    if cfg.model_size not in MODEL_SIZE_PRESETS:
        raise ValueError(f"Unknown model_size={cfg.model_size!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    out_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = out_dir / f"{cfg.run_id}.pickle"

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Minimal BenchmarkConfig-shaped object for build_model / make_batch.
    from cs336_systems.e2e_timing.e2e import BenchmarkConfig

    bcfg = BenchmarkConfig(
        model_size=cfg.model_size,
        mode="train",
        vocab_size=cfg.vocab_size,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        seed=cfg.seed,
        mixed_precision=cfg.mixed_precision,
        do_optimizer=True,
    )
    device = torch.device("cuda")
    model = build_model(bcfg, device)
    model.train()
    optimizer = AdamW(model.parameters())
    x, y = make_batch(bcfg, device)
    _sync()

    after_init = _mem_snapshot()
    _warmup(
        model,
        optimizer,
        x,
        y,
        mode=cfg.mode,
        mixed_precision=cfg.mixed_precision,
        n=cfg.warmup,
    )
    # Drop warmup peaks; keep weights / Adam state resident.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    after_warmup = _mem_snapshot()

    stages: dict[str, Any] = {}
    boundaries: list[dict[str, Any]] = []
    overall_peak_allocated = after_warmup["allocated_bytes"]

    amp = _amp_context(cfg.mixed_precision)

    with _record_history():
        _sync()
        baseline_allocated = int(torch.cuda.memory_allocated())
        boundaries.append(
            {
                "name": "start",
                "time_us": _now_us(),
                "allocated_bytes": baseline_allocated,
            }
        )
        torch.cuda.reset_peak_memory_stats()

        if cfg.mode == "forward":
            with amp:
                logits = model(x)
            _sync()
            snap = _mem_snapshot()
            stages["forward"] = snap
            overall_peak_allocated = max(overall_peak_allocated, snap["max_allocated_bytes"])
            boundaries.append({"name": "forward", "time_us": _now_us()})
            del logits
        else:
            optimizer.zero_grad(set_to_none=True)
            _sync()
            torch.cuda.reset_peak_memory_stats()

            with amp:
                logits = model(x)
            _sync()
            snap = _mem_snapshot()
            stages["forward"] = snap
            overall_peak_allocated = max(overall_peak_allocated, snap["max_allocated_bytes"])
            boundaries.append({"name": "forward", "time_us": _now_us()})
            torch.cuda.reset_peak_memory_stats()

            with amp:
                loss = cross_entropy(logits, y)
            _sync()
            snap = _mem_snapshot()
            stages["loss"] = snap
            overall_peak_allocated = max(overall_peak_allocated, snap["max_allocated_bytes"])
            boundaries.append({"name": "loss", "time_us": _now_us()})
            torch.cuda.reset_peak_memory_stats()

            loss.backward()
            _sync()
            snap = _mem_snapshot()
            stages["backward"] = snap
            overall_peak_allocated = max(overall_peak_allocated, snap["max_allocated_bytes"])
            boundaries.append({"name": "backward", "time_us": _now_us()})
            torch.cuda.reset_peak_memory_stats()

            optimizer.step()
            _sync()
            snap = _mem_snapshot()
            stages["optimizer"] = snap
            overall_peak_allocated = max(overall_peak_allocated, snap["max_allocated_bytes"])
            boundaries.append({"name": "optimizer", "time_us": _now_us()})

        torch.cuda.memory._dump_snapshot(str(pickle_path))
        boundaries.append({"name": "end", "time_us": _now_us()})

    hparams = dict(MODEL_SIZE_PRESETS[cfg.model_size])
    result: dict[str, Any] = {
        "config": asdict(cfg),
        "model_hparams": hparams,
        "oom": False,
        "pickle_path_abs": str(pickle_path),
        "after_init": after_init,
        "after_warmup": after_warmup,
        "baseline_allocated_bytes": int(baseline_allocated),
        "stages": stages,
        "stage_boundaries": boundaries,
        "peak_allocated_bytes": int(overall_peak_allocated),
        "peak_allocated_gib": overall_peak_allocated / (1024**3),
        "peak_allocated_mib": overall_peak_allocated / (1024**2),
    }
    return result


def try_profile_cell(cfg: MemoryCellConfig, out_dir: Path) -> dict[str, Any]:
    try:
        return profile_cell(cfg, out_dir)
    except Exception as err:  # noqa: BLE001
        msg = str(err).lower()
        oom = "out of memory" in msg or "cuda oom" in msg
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "config": asdict(cfg),
            "oom": oom,
            "error": str(err),
            "stages": {},
            "stage_boundaries": [],
            "peak_allocated_bytes": None,
            "peak_allocated_gib": None,
        }
