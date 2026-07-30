#!/usr/bin/env python3
# =============================================================================
# Problem (naive_ddp_benchmarking): Naive DDP Benchmarking
# =============================================================================
#
# Architecture:
#   Driver mode (no --worker flag): CPU-only scheduler. Spawns a fresh
#     subprocess for *every* configuration.  The driver never touches CUDA,
#     so process exit is the only CUDA-cleanup mechanism we need.
#
#   Worker mode (--worker MODE BS): Runs ONE (mode, batch_size) pair and
#     appends results to the shared CSV. Exits immediately after —
#     the OS reclaims all GPU memory.
#
# This two-level design avoids CUDA-context / NCCL-IPC contamination between
# single-GPU and DDP runs within the same Python process lifetime.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import subprocess
import sys
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

# Empirical OOM limits for xl model in fp32 + AdamW on 97 GB GPU:
#
#   AdamW allocates m + v (2× params = 27.2 GB) lazily during first optimizer.step().
#   After init, persistent baseline = params(13.6) + m(13.6) + v(13.6) = 40.9 GB.
#   Forward activations for batch=4 ≈ 29 GB, for batch=8 ≈ 57.8 GB.
#
#   Single-GPU: 40.9 + 29 = 70 GB ✓,  40.9 + 57.8 = 98.7 GB ✗ OOM
#   DDP per=2:  40.9 + 14.5 = 55 GB ✓
#   DDP per=4:  40.9 + 29 = 70 GB ✓
#   DDP per=8:  40.9 + 57.8 = 98.7 GB ✗ OOM
SINGLE_BATCH_LIMIT = 4
DDP_BATCH_LIMIT = 8

XL_HPARAMS = dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)


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
# Helpers (used only in worker processes)
# ---------------------------------------------------------------------------
def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_model(device: torch.device) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        **XL_HPARAMS,
    ).to(device)


def _make_batch(bs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, VOCAB_SIZE, (bs, CONTEXT_LENGTH), device=device)
    y = torch.randint(0, VOCAB_SIZE, (bs, CONTEXT_LENGTH), device=device)
    return x, y


def _timed_train_step(model, optimizer, x, y, device, *, do_gradient_sync=False):
    optimizer.zero_grad(set_to_none=True)

    _sync(device); t0 = timeit.default_timer()
    logits = model(x)
    _sync(device); fwd = timeit.default_timer() - t0

    _sync(device); t0 = timeit.default_timer()
    loss = cross_entropy(logits, y)
    _sync(device); loss_s = timeit.default_timer() - t0

    _sync(device); t0 = timeit.default_timer()
    loss.backward()
    _sync(device); bwd = timeit.default_timer() - t0

    if do_gradient_sync:
        _sync(device); t0 = timeit.default_timer()
        model.finish_gradient_synchronization()
        _sync(device); sync_s = timeit.default_timer() - t0
    else:
        sync_s = 0.0

    _sync(device); t0 = timeit.default_timer()
    optimizer.step()
    _sync(device); opt_s = timeit.default_timer() - t0

    del logits, loss
    return {"forward": fwd, "loss": loss_s, "backward": bwd,
            "gradient_sync": sync_s, "optimizer": opt_s}


# ---------------------------------------------------------------------------
# Per-parameter all_reduce recording (NaiveDDP only)
# ---------------------------------------------------------------------------
def _record_per_param(ddp_model, x, y, optimizer, device, csv_path):
    optimizer.zero_grad(set_to_none=True)
    _ = ddp_model(x)
    loss = cross_entropy(_, y)
    loss.backward()
    del _, loss

    inner = ddp_model.module
    world_size = dist.get_world_size()
    results = []

    _sync(device)
    for p_name, param in inner.named_parameters():
        if param.grad is None:
            continue
        _sync(device); t0 = timeit.default_timer()
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        _sync(device); dt = timeit.default_timer() - t0
        param.grad.div_(world_size)
        results.append((p_name, param.numel(), dt))

    optimizer.step()

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param_name", "numel", "bytes", "latency_s"])
        for name, n, lat in results:
            w.writerow([name, n, n * 4, lat])
    print(f"  per-param data saved ({len(results)} tensors)")


# ---------------------------------------------------------------------------
# Worker: single-GPU run (ONE batch size, exits after writing CSV)
# ---------------------------------------------------------------------------
def _worker_single(batch_size: int, output_csv: str) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = _build_model(device)
    model.train()
    optimizer = AdamW(model.parameters())
    x, y = _make_batch(batch_size, device)
    records = []

    for i in range(WARMUP_STEPS):
        _timed_train_step(model, optimizer, x, y, device, do_gradient_sync=False)

    for i in range(MEASURE_STEPS):
        segs = _timed_train_step(model, optimizer, x, y, device, do_gradient_sync=False)
        total = sum(segs.values())
        records.append(StepRecord(
            mode="single", batch_size=batch_size, step=i, rank=0,
            forward_s=segs["forward"], loss_s=segs["loss"],
            backward_s=segs["backward"], gradient_sync_s=0.0,
            optimizer_s=segs["optimizer"], total_s=total,
        ))
        print(f"  step {i + 1:>2}/{MEASURE_STEPS}  fwd={segs['forward']:.3f}s  "
              f"loss={segs['loss']:.3f}s  bwd={segs['backward']:.3f}s  "
              f"opt={segs['optimizer']:.3f}s  sum={total:.3f}s")

    _append_csv(Path(output_csv), records)
    _print_summary(records)


# ---------------------------------------------------------------------------
# Worker: DDP run (ONE batch size, uses mp.spawn internally)
# ---------------------------------------------------------------------------
def _ddp_worker(rank, world_size, batch_size, master_port, output_csv, per_param_csv):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    try:
        model = _build_model(device); model.train()
        ddp_model = NaiveDDP(model)
        optimizer = AdamW(ddp_model.parameters())
        per_rank_batch = batch_size // world_size
        x, y = _make_batch(per_rank_batch, device)
        local_records = []

        for _ in range(WARMUP_STEPS):
            _timed_train_step(ddp_model, optimizer, x, y, device, do_gradient_sync=True)

        for i in range(MEASURE_STEPS):
            segs = _timed_train_step(ddp_model, optimizer, x, y, device, do_gradient_sync=True)
            total = sum(segs.values())
            local_records.append(StepRecord(
                mode="naive_ddp", batch_size=batch_size, step=i, rank=rank,
                forward_s=segs["forward"], loss_s=segs["loss"],
                backward_s=segs["backward"],
                gradient_sync_s=segs["gradient_sync"],
                optimizer_s=segs["optimizer"], total_s=total,
            ))
            if rank == 0:
                print(f"  step {i + 1:>2}/{MEASURE_STEPS}  fwd={segs['forward']:.3f}s  "
                      f"loss={segs['loss']:.3f}s  bwd={segs['backward']:.3f}s  "
                      f"sync={segs['gradient_sync']:.3f}s  opt={segs['optimizer']:.3f}s  "
                      f"sum={total:.3f}s")

        if per_param_csv is not None and rank == 0:
            print("  recording per-parameter all-reduce latencies …")
            _record_per_param(ddp_model, x, y, optimizer, device, per_param_csv)
        elif per_param_csv is not None:
            _timed_train_step(ddp_model, optimizer, x, y, device, do_gradient_sync=True)

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_records)
        if rank == 0:
            all_recs = [r for rlist in gathered if rlist is not None for r in rlist]
            _append_csv(Path(output_csv), all_recs)
    finally:
        dist.barrier(); dist.destroy_process_group()


def _worker_ddp(batch_size: int, output_csv: str, master_port: int,
                per_param_csv: str | None) -> None:
    mp.spawn(_ddp_worker,
             args=(2, batch_size, str(master_port), output_csv, per_param_csv),
             nprocs=2, join=True)
    recs = _load_records(Path(output_csv), mode="naive_ddp", batch_size=batch_size)
    _print_summary(recs)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _append_csv(path: Path, records: list[StepRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        if write_header: w.writeheader()
        for r in records: w.writerow(asdict(r))


def _load_records(path, *, mode=None, batch_size=None):
    recs = []
    if not path.exists(): return recs
    with path.open("r") as f:
        for row in csv.DictReader(f):
            if mode and row["mode"] != mode: continue
            if batch_size and int(row["batch_size"]) != batch_size: continue
            recs.append(StepRecord(
                mode=row["mode"], batch_size=int(row["batch_size"]),
                step=int(row["step"]), rank=int(row["rank"]),
                forward_s=float(row["forward_s"]), loss_s=float(row["loss_s"]),
                backward_s=float(row["backward_s"]),
                gradient_sync_s=float(row["gradient_sync_s"]),
                optimizer_s=float(row["optimizer_s"]),
                total_s=float(row["total_s"])))
    return recs


def _print_summary(records):
    if not records: return
    m, bs = records[0].mode, records[0].batch_size
    print(f"\n  Summary [{m}] batch={bs} ({len(records)} steps):")
    for key, label in [("forward_s","forward"), ("loss_s","loss"),
                        ("backward_s","backward"), ("gradient_sync_s","grad_sync"),
                        ("optimizer_s","optimizer"), ("total_s","TOTAL")]:
        vals = [getattr(r, key) for r in records]
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        if label == "grad_sync" and mu == 0.0: continue
        tot = statistics.mean([r.total_s for r in records])
        pct = (mu / tot * 100) if tot > 0 else 0.0
        print(f"    {label:>12}: {mu:.4f}s ± {sd:.4f}s  ({pct:.1f}%)")


# ---------------------------------------------------------------------------
# Driver: CPU-only, launches subprocess for each config
# ---------------------------------------------------------------------------
def _driver(args: argparse.Namespace) -> None:
    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists(): output_csv.unlink()
    if PER_PARAM_CSV.exists(): PER_PARAM_CSV.unlink()

    # Prevent CUDA memory fragmentation from causing spurious OOMs on
    # models with heterogeneous tensor sizes (291 nn.Parameter tensors
    # ranging from 10 KB to 105 MB).
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    python = sys.executable
    script = __file__

    for i, bs in enumerate(args.batch_sizes):
        port = args.master_port + i

        # --- Single-GPU ---
        if not args.skip_single and bs <= SINGLE_BATCH_LIMIT:
            print(f"\n── Single-GPU · batch={bs} ──", flush=True)
            subprocess.run(
                [python, script,
                 "--worker", "single", str(bs),
                 "--output", str(output_csv)],
                check=True, env=env,
            )

        # --- DDP ---
        if not args.skip_ddp and bs <= DDP_BATCH_LIMIT:
            print(f"\n── NaiveDDP 2-GPU · batch={bs} ──", flush=True)
            cmd = [python, script,
                   "--worker", "ddp", str(bs),
                   "--output", str(output_csv),
                   "--master-port", str(port)]
            if not args.no_per_param and i == 0:
                cmd += ["--per-param-csv", str(PER_PARAM_CSV)]
            subprocess.run(cmd, check=True, env=env)

    print(f"\n✓ All results → {output_csv}")
    if PER_PARAM_CSV.exists():
        print(f"✓ Per-param latencies → {PER_PARAM_CSV}")


# ---------------------------------------------------------------------------
# CLI
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
    p.add_argument("--no-per-param", action="store_true")
    # Worker mode
    p.add_argument("--worker", nargs=2, metavar=("MODE", "BS"),
                   help="Internal: run single config and exit. MODE ∈ {single, ddp}")
    p.add_argument("--per-param-csv", default=None,
                   help="Internal: path for per-param CSV (ddp worker only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.worker is not None:
        # ── Worker process (touches CUDA) ──
        mode, bs_str = args.worker
        bs = int(bs_str)
        if mode == "single":
            _worker_single(bs, args.output)
        elif mode == "ddp":
            _worker_ddp(bs, args.output, args.master_port, args.per_param_csv)
        else:
            raise ValueError(f"Unknown worker mode: {mode}")
    else:
        # ── Driver process (CPU only) ──
        _driver(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
