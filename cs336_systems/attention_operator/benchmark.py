"""Single-cell benchmark for scaled dot-product attention."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from cs336_basics.model import scaled_dot_product_attention

DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP = 5
DEFAULT_ITERS = 100
DEFAULT_SEED = 42


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    batch_size: int = DEFAULT_BATCH_SIZE
    seq_len: int = 256
    d_model: int = 16
    warmup: int = DEFAULT_WARMUP
    iters: int = DEFAULT_ITERS
    seed: int = DEFAULT_SEED


@dataclass
class BenchmarkResult:
    batch_size: int
    seq_len: int
    d_model: int
    forward_ms: float | None
    backward_ms: float | None
    memory_before_backward_bytes: int | None
    memory_before_backward_gib: float | None
    oom: bool
    error: str | None

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


def _make_qkv(cfg: AttentionBenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (cfg.batch_size, cfg.seq_len, cfg.d_model)
    q = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    return q, k, v


def _step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    out = scaled_dot_product_attention(Q=q, K=k, V=v, mask=None)
    return out.sum()


def benchmark_cell(cfg: AttentionBenchmarkConfig, device: torch.device | None = None) -> BenchmarkResult:
    """Run paired forward/backward benchmark for one (B, S, d) configuration."""
    if device is None:
        device = torch.device("cuda")

    if device.type != "cuda":
        raise RuntimeError("CUDA required")

    _cleanup(device)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    base = dict(
        batch_size=cfg.batch_size,
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
    )

    try:
        for _ in range(cfg.warmup):
            q, k, v = _make_qkv(cfg, device)
            out = _step(q, k, v)
            out.backward()
            q.grad = None
            k.grad = None
            v.grad = None
            del q, k, v, out
            _cleanup(device)

        forward_total = 0.0
        backward_total = 0.0
        mem_before_bwd: int | None = None

        for i in range(cfg.iters):
            q, k, v = _make_qkv(cfg, device)

            t0 = time.perf_counter()
            out = _step(q, k, v)
            _sync(device)
            t1 = time.perf_counter()
            forward_total += t1 - t0

            if i == 0:
                mem_before_bwd = int(torch.cuda.memory_allocated())

            t2 = time.perf_counter()
            out.backward()
            _sync(device)
            t3 = time.perf_counter()
            backward_total += t3 - t2

            q.grad = None
            k.grad = None
            v.grad = None
            del q, k, v, out

        return BenchmarkResult(
            **base,
            forward_ms=forward_total / cfg.iters * 1000.0,
            backward_ms=backward_total / cfg.iters * 1000.0,
            memory_before_backward_bytes=mem_before_bwd,
            memory_before_backward_gib=mem_before_bwd / (1024**3) if mem_before_bwd is not None else None,
            oom=False,
            error=None,
        )
    except torch.cuda.OutOfMemoryError as err:
        return BenchmarkResult(
            **base,
            forward_ms=None,
            backward_ms=None,
            memory_before_backward_bytes=None,
            memory_before_backward_gib=None,
            oom=True,
            error=str(err),
        )
    finally:
        _cleanup(device)
