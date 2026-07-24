"""Grid sweep over (d_model, seq_len) for attention operator benchmark."""

from __future__ import annotations

from typing import Iterable

import torch

from cs336_systems.attention_operator.benchmark import (
    AttentionBenchmarkConfig,
    BenchmarkResult,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ITERS,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
    benchmark_cell,
)

DEFAULT_D_VALUES: tuple[int, ...] = (16, 32, 64, 128)
DEFAULT_S_VALUES: tuple[int, ...] = (256, 1024, 4096, 8192, 16384, 24576, 32768)
EXTRA_S_IF_NO_OOM: tuple[int, ...] = (40960, 49152)


def iter_grid(
    d_values: Iterable[int] = DEFAULT_D_VALUES,
    s_values: Iterable[int] = DEFAULT_S_VALUES,
) -> list[tuple[int, int]]:
    return [(d, s) for d in d_values for s in s_values]


def run_sweep(
    *,
    d_values: tuple[int, ...] = DEFAULT_D_VALUES,
    s_values: tuple[int, ...] = DEFAULT_S_VALUES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup: int = DEFAULT_WARMUP,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    device: torch.device | None = None,
    extend_for_oom: bool = True,
) -> list[BenchmarkResult]:
    if device is None:
        device = torch.device("cuda")

    results: list[BenchmarkResult] = []
    grid = iter_grid(d_values, s_values)

    for d, s in grid:
        cfg = AttentionBenchmarkConfig(
            batch_size=batch_size,
            seq_len=s,
            d_model=d,
            warmup=warmup,
            iters=iters,
            seed=seed,
        )
        print(f"  d={d} S={s} …", flush=True)
        result = benchmark_cell(cfg, device)
        if result.oom:
            print("    OOM", flush=True)
        else:
            print(
                f"    fwd {result.forward_ms:.2f} ms · bwd {result.backward_ms:.2f} ms · "
                f"mem {result.memory_before_backward_gib:.3f} GiB",
                flush=True,
            )
        results.append(result)

    if extend_for_oom and not any(r.oom for r in results):
        print("── No OOM in base grid; extending S for d=128 ──", flush=True)
        for s in EXTRA_S_IF_NO_OOM:
            cfg = AttentionBenchmarkConfig(
                batch_size=batch_size,
                seq_len=s,
                d_model=128,
                warmup=warmup,
                iters=iters,
                seed=seed,
            )
            print(f"  d=128 S={s} …", flush=True)
            result = benchmark_cell(cfg, device)
            if result.oom:
                print("    OOM", flush=True)
            else:
                print(
                    f"    fwd {result.forward_ms:.2f} ms · bwd {result.backward_ms:.2f} ms · "
                    f"mem {result.memory_before_backward_gib:.3f} GiB",
                    flush=True,
                )
            results.append(result)
            if result.oom:
                break

    return results
