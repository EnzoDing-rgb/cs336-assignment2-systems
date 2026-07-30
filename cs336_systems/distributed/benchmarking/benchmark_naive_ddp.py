# =============================================================================
# Problem (naive_ddp_benchmarking): Naive DDP Benchmarking
# =============================================================================
#
# Measures:
#   (a) Single-GPU baseline (no DDP, pure local training)
#   (b) 2-GPU NaiveDDP (per-parameter gradient all-reduce, ~291 calls per step)
#   (c) 2-GPU FlattenDDP (single batched all-reduce for all gradients)
#
# Each configuration runs in its own subprocess so GPU memory is cleanly
# released by the OS on process exit — no cross-run leaks.
#
# Sweeps batch sizes [4, 8, 16, 32, 64] for the xl model (Section 2.1.2).
# Timed segments per step: forward | loss | backward | gradient_sync | optimizer.
#
# Output:
#   artifacts/naive_ddp_benchmark.csv           — per-step segment timings
#   artifacts/naive_ddp_per_param_latency.csv   — per-parameter all-reduce latency
# =============================================================================

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import timeit
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from cs336_systems.distributed.naive_ddp import NaiveDDP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUT_CSV = ARTIFACTS_DIR / "naive_ddp_benchmark.csv"
PER_PARAM_CSV = ARTIFACTS_DIR / "naive_ddp_per_param_latency.csv"

BATCH_SIZES = (4, 8, 16, 32, 64)
WARMUP_STEPS = 5
MEASURE_STEPS = 10
VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 512
SEED = 0

XL_HPARAMS = {
    "d_model": 2560,
    "d_ff": 10240,
    "num_layers": 32,
    "num_heads": 32,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    mode: str
    batch_size: int
    step: int
    rank: int
    forward_s: float
    loss_s: float
    backward_s: float
    gradient_sync_s: float
    optimizer_s: float
    total_s: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_model(device: torch.device) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        **XL_HPARAMS,
    )
    return model.to(device)


def _make_batch(
    batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, VOCAB_SIZE, (batch_size, CONTEXT_LENGTH), device=device)
    y = torch.randint(0, VOCAB_SIZE, (batch_size, CONTEXT_LENGTH), device=device)
    return x, y


def _timed_train_step(
    model: BasicsTransformerLM | NaiveDDP,
    optimizer: AdamW,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    *,
    do_gradient_sync: bool = False,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)

    _sync(device)
    t0 = timeit.default_timer()
    logits = model(x)
    _sync(device)
    forward_s = timeit.default_timer() - t0

    _sync(device)
    t0 = timeit.default_timer()
    loss = cross_entropy(logits, y)
    _sync(device)
    loss_s = timeit.default_timer() - t0

    _sync(device)
    t0 = timeit.default_timer()
    loss.backward()
    _sync(device)
    backward_s = timeit.default_timer() - t0

    if do_gradient_sync:
        _sync(device)
        t0 = timeit.default_timer()
        model.finish_gradient_synchronization()
        _sync(device)
        gradient_sync_s = timeit.default_timer() - t0
    else:
        gradient_sync_s = 0.0

    _sync(device)
    t0 = timeit.default_timer()
    optimizer.step()
    _sync(device)
    optimizer_s = timeit.default_timer() - t0

    return {
        "forward": forward_s,
        "loss": loss_s,
        "backward": backward_s,
        "gradient_sync": gradient_sync_s,
        "optimizer": optimizer_s,
    }


def _measure_per_param_all_reduce(
    model: NaiveDDP,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: AdamW,
    device: torch.device,
) -> list[tuple[str, int, float]]:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = cross_entropy(logits, y)
    loss.backward()

    inner = model.module
    world_size = dist.get_world_size()
    results: list[tuple[str, int, float]] = []

    _sync(device)
    for p_name, param in inner.named_parameters():
        if param.grad is None:
            continue
        _sync(device)
        t0 = timeit.default_timer()
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        _sync(device)
        dt = timeit.default_timer() - t0
        param.grad.div_(world_size)
        results.append((p_name, param.numel(), dt))

    optimizer.step()
    return results


# ---------------------------------------------------------------------------
# Single-GPU worker (runs in its own process)
# ---------------------------------------------------------------------------
def _single_gpu_worker(
    batch_size: int,
    output_csv: str,
) -> None:
    """Run single-GPU benchmark in an isolated process."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = _build_model(device)
    model.train()
    optimizer = AdamW(model.parameters())
    x, y = _make_batch(batch_size, device)
    records: list[StepRecord] = []

    # Warmup
    for i in range(WARMUP_STEPS):
        _timed_train_step(model, optimizer, x, y, device, do_gradient_sync=False)
        print(f"  warmup {i + 1}/{WARMUP_STEPS}")

    # Measurement
    for i in range(MEASURE_STEPS):
        segs = _timed_train_step(model, optimizer, x, y, device, do_gradient_sync=False)
        total = sum(segs.values())
        rec = StepRecord(
            mode="single", batch_size=batch_size, step=i, rank=0,
            forward_s=segs["forward"], loss_s=segs["loss"],
            backward_s=segs["backward"], gradient_sync_s=0.0,
            optimizer_s=segs["optimizer"], total_s=total,
        )
        records.append(rec)
        print(
            f"  step {i + 1:>2}/{MEASURE_STEPS}  "
            f"fwd={segs['forward']:.3f}s  loss={segs['loss']:.3f}s  "
            f"bwd={segs['backward']:.3f}s  opt={segs['optimizer']:.3f}s  "
            f"sum={total:.3f}s"
        )

    _append_csv(Path(output_csv), records)
    # Process exit releases all GPU memory automatically.


def run_single_gpu(batch_size: int, output_csv: Path) -> list[StepRecord]:
    """Launch single-GPU benchmark in a subprocess; return loaded records."""
    print(f"\n── Single-GPU · batch={batch_size} ──", flush=True)
    ctx = mp.get_context("spawn")
    proc = ctx.Process(
        target=_single_gpu_worker,
        args=(batch_size, str(output_csv)),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(
            f"Single-GPU worker (batch={batch_size}) exited with code {proc.exitcode}"
        )
    return _load_records(output_csv, mode="single", batch_size=batch_size)


# ---------------------------------------------------------------------------
# DDP worker + launcher
# ---------------------------------------------------------------------------
def _ddp_worker(
    rank: int,
    world_size: int,
    batch_size: int,
    master_port: str,
    output_csv: str,
    per_param_csv: str | None,
    ddp_mode: str = "naive",
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    try:
        model = _build_model(device)
        model.train()

        if ddp_mode == "flatten":
            from cs336_systems.distributed.flatten_ddp import FlattenDDP
            ddp_model = FlattenDDP(model)
        else:
            ddp_model = NaiveDDP(model)

        optimizer = AdamW(ddp_model.parameters())
        per_rank_batch = batch_size // world_size
        x, y = _make_batch(per_rank_batch, device)
        local_records: list[StepRecord] = []

        # Warmup
        for i in range(WARMUP_STEPS):
            _timed_train_step(ddp_model, optimizer, x, y, device, do_gradient_sync=True)
            if rank == 0:
                print(f"  warmup {i + 1}/{WARMUP_STEPS}")

        # Measurement
        for i in range(MEASURE_STEPS):
            segs = _timed_train_step(ddp_model, optimizer, x, y, device, do_gradient_sync=True)
            total = sum(segs.values())
            rec = StepRecord(
                mode=f"{ddp_mode}_ddp", batch_size=batch_size, step=i, rank=rank,
                forward_s=segs["forward"], loss_s=segs["loss"],
                backward_s=segs["backward"],
                gradient_sync_s=segs["gradient_sync"],
                optimizer_s=segs["optimizer"], total_s=total,
            )
            local_records.append(rec)
            if rank == 0:
                print(
                    f"  step {i + 1:>2}/{MEASURE_STEPS}  "
                    f"fwd={segs['forward']:.3f}s  loss={segs['loss']:.3f}s  "
                    f"bwd={segs['backward']:.3f}s  sync={segs['gradient_sync']:.3f}s  "
                    f"opt={segs['optimizer']:.3f}s  sum={total:.3f}s"
                )

        # Per-parameter timing (separate step, rank 0 only)
        if per_param_csv is not None and rank == 0 and ddp_mode == "naive":
            print("  recording per-parameter all_reduce latencies …")
            pp_data = _measure_per_param_all_reduce(ddp_model, x, y, optimizer, device)
            pp_path = Path(per_param_csv)
            pp_path.parent.mkdir(parents=True, exist_ok=True)
            with pp_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["param_name", "numel", "bytes", "latency_s"])
                for p_name, numel, lat in pp_data:
                    writer.writerow([p_name, numel, numel * 4, lat])
            print(f"  per-param data saved ({len(pp_data)} tensors)")
        elif per_param_csv is not None:
            _timed_train_step(ddp_model, optimizer, x, y, device, do_gradient_sync=True)

        # Gather
        gathered: list[list[StepRecord] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_records)

        if rank == 0:
            all_records: list[StepRecord] = []
            for rrecs in gathered:
                if rrecs is not None:
                    all_records.extend(rrecs)
            _append_csv(Path(output_csv), all_records)

    finally:
        dist.barrier()
        dist.destroy_process_group()
        # Process exit releases memory.


def run_ddp(
    batch_size: int,
    output_csv: Path,
    master_port: int,
    *,
    ddp_mode: str = "naive",
    record_per_param: bool = False,
) -> list[StepRecord]:
    world_size = 2
    per_rank = batch_size // world_size
    label = {"naive": "NaiveDDP", "flatten": "FlattenDDP"}[ddp_mode]
    print(f"\n── {label} 2-GPU · total_batch={batch_size} (per_rank={per_rank}) · port={master_port} ──", flush=True)

    pp_csv = str(PER_PARAM_CSV) if (record_per_param and ddp_mode == "naive") else None

    mp.spawn(
        fn=_ddp_worker,
        args=(world_size, batch_size, str(master_port), str(output_csv), pp_csv, ddp_mode),
        nprocs=world_size,
        join=True,
    )

    return _load_records(output_csv, mode=f"{ddp_mode}_ddp", batch_size=batch_size)


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------
def _append_csv(path: Path, records: list[StepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(asdict(records[0]).keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def _load_records(
    path: Path, *, mode: str | None = None, batch_size: int | None = None
) -> list[StepRecord]:
    records: list[StepRecord] = []
    with path.open("r") as f:
        for row in csv.DictReader(f):
            if mode and row["mode"] != mode:
                continue
            if batch_size and int(row["batch_size"]) != batch_size:
                continue
            records.append(StepRecord(
                mode=row["mode"],
                batch_size=int(row["batch_size"]),
                step=int(row["step"]),
                rank=int(row["rank"]),
                forward_s=float(row["forward_s"]),
                loss_s=float(row["loss_s"]),
                backward_s=float(row["backward_s"]),
                gradient_sync_s=float(row["gradient_sync_s"]),
                optimizer_s=float(row["optimizer_s"]),
                total_s=float(row["total_s"]),
            ))
    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(records: list[StepRecord]) -> None:
    if not records:
        return
    mode = records[0].mode
    bs = records[0].batch_size
    keys = ["forward_s", "loss_s", "backward_s", "gradient_sync_s", "optimizer_s", "total_s"]
    labels = ["forward", "loss", "backward", "grad_sync", "optimizer", "TOTAL"]
    print(f"\n  Summary [{mode}] batch={bs} ({len(records)} steps from all ranks):")
    for key, label in zip(keys, labels):
        vals = [getattr(r, key) for r in records]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if label == "grad_sync" and mean == 0.0:
            continue
        total_mean = statistics.mean([r.total_s for r in records])
        pct = (mean / total_mean * 100) if total_mean > 0 else 0.0
        print(f"    {label:>12}: {mean:.4f}s ± {std:.4f}s  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Naive DDP Benchmark — CS336 Assignment 2")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    p.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    p.add_argument("--steps", type=int, default=MEASURE_STEPS)
    p.add_argument("--master-port", type=int, default=29505)
    p.add_argument("--output", default=str(OUTPUT_CSV))
    p.add_argument("--skip-single", action="store_true")
    p.add_argument("--skip-ddp", action="store_true")
    p.add_argument("--ddp-mode", choices=("naive", "flatten"), default="naive",
                   help="DDP strategy: naive (per-param all-reduce) or flatten (single batched).")
    p.add_argument("--no-per-param", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()
    if PER_PARAM_CSV.exists():
        PER_PARAM_CSV.unlink()

    for i, bs in enumerate(args.batch_sizes):
        port = args.master_port + i

        # ---- Single-GPU baseline (subprocess) ----
        if not args.skip_single:
            try:
                records = run_single_gpu(bs, output_csv)
                print_summary(records)
            except RuntimeError as e:
                if "OOM" in str(e) or "out of memory" in str(e).lower():
                    print(f"  ✗ Single-GPU batch={bs}: OOM ({e})")
                else:
                    raise

        # ---- 2-GPU DDP ----
        if not args.skip_ddp:
            do_pp = (not args.no_per_param) and (i == 0) and (args.ddp_mode == "naive")
            try:
                records = run_ddp(
                    bs, output_csv, master_port=port,
                    ddp_mode=args.ddp_mode, record_per_param=do_pp,
                )
                print_summary(records)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "OOM" in str(e) or "out of memory" in str(e).lower():
                    print(f"  ✗ NaiveDDP batch={bs}: OOM ({e})")
                else:
                    raise

    print(f"\n✓ All results → {output_csv}")
    if PER_PARAM_CSV.exists():
        print(f"✓ Per-param latencies → {PER_PARAM_CSV}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
