"""
FSDP benchmark: peak memory + step timing.

Configs: 2-GPU × {FP32, BF16} + 4-GPU × {FP32, BF16}.
xl model, batch=4 total (per_rank=batch/2 or batch/4).

Output: artifacts/fsdp_benchmark.csv
"""

from __future__ import annotations

import csv
import gc
import os
import statistics
import timeit
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.distributed.fsdp import FSDP
from cs336_systems.distributed.sharded_optimizer import ShardedOptimizer

WARMUP = 5
STEPS  = 10
VOCAB  = 10_000
CTX    = 512
SEED   = 0
XL     = {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
OUT    = Path("artifacts/fsdp_benchmark.csv")


def _sync(d: torch.device) -> None:
    if d.type == "cuda":
        torch.cuda.synchronize(d)


def _mem_mib(d: torch.device) -> float:
    if d.type != "cuda":
        return 0.0
    return torch.cuda.memory_allocated(d) / (1024 * 1024)


def _peak_mib_since_reset(d: torch.device) -> float:
    if d.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(d) / (1024 * 1024)


def _reset_peak(d: torch.device) -> None:
    if d.type == "cuda":
        torch.cuda.reset_peak_memory_stats(d)


def _worker(rank, world_size, mode, port):
    """mode: 'fsdp_fp32' | 'fsdp_bf16'"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    compute_dtype = torch.bfloat16 if mode == "fsdp_bf16" else None
    dtype_label = "bf16" if compute_dtype is not None else "fp32"
    use_autocast = compute_dtype is not None

    _reset_peak(dev)
    model = BasicsTransformerLM(vocab_size=VOCAB, context_length=CTX, **XL).to(dev)
    model.train()
    fsdp = FSDP(model, compute_dtype=compute_dtype)
    mem_after_init = _mem_mib(dev)

    _reset_peak(dev)
    opt = ShardedOptimizer(fsdp.parameters(), AdamW, lr=1e-3)
    mem_after_opt_ctor = _mem_mib(dev)

    per_rank_batch = 4 // world_size
    x = torch.randint(0, VOCAB, (per_rank_batch, CTX), device=dev)
    y = torch.randint(0, VOCAB, (per_rank_batch, CTX), device=dev)

    # Warmup
    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = fsdp(x)
        else:
            logits = fsdp(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        fsdp.finish_gradient_synchronization()
        opt.step()
        _sync(dev)

    # Measurement
    rows = []
    for step_idx in range(STEPS):
        _reset_peak(dev)
        opt.zero_grad(set_to_none=True)

        _sync(dev); t0 = timeit.default_timer()
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = fsdp(x)
        else:
            logits = fsdp(x)
        _sync(dev); fwd = timeit.default_timer() - t0

        _sync(dev); t0 = timeit.default_timer()
        loss = cross_entropy(logits, y)
        _sync(dev); loss_t = timeit.default_timer() - t0

        _sync(dev); t0 = timeit.default_timer()
        loss.backward()
        _sync(dev); bwd = timeit.default_timer() - t0

        mem_before_sync = _mem_mib(dev)

        _sync(dev); t0 = timeit.default_timer()
        fsdp.finish_gradient_synchronization()
        _sync(dev); sync_t = timeit.default_timer() - t0

        mem_before_opt = _mem_mib(dev)
        peak_before_opt = _peak_mib_since_reset(dev)

        _reset_peak(dev)
        _sync(dev); t0 = timeit.default_timer()
        opt.step()
        _sync(dev); opt_t = timeit.default_timer() - t0

        mem_after_opt = _mem_mib(dev)
        peak_during_opt = _peak_mib_since_reset(dev)

        total = fwd + loss_t + bwd + sync_t + opt_t
        rows.append((f"fsdp_{dtype_label}", world_size, per_rank_batch,
                     step_idx, rank,
                     fwd, loss_t, bwd, sync_t, opt_t, total,
                     mem_after_init, mem_after_opt_ctor,
                     mem_before_sync, mem_before_opt, peak_before_opt,
                     mem_after_opt, peak_during_opt))

        if rank == 0:
            print(f"  step {step_idx+1:>2}/{STEPS}  fwd={fwd:.3f}s  bwd={bwd:.3f}s  "
                  f"sync={sync_t:.3f}s  opt={opt_t:.3f}s  total={total:.3f}s  "
                  f"peak_before_opt={peak_before_opt:.0f}MiB")

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUT.exists()
        with OUT.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "mode","world_size","per_rank_batch","step","rank",
                    "forward_s","loss_s","backward_s","gradient_sync_s","optimizer_s","total_s",
                    "mem_after_init_mib","mem_after_opt_ctor_mib",
                    "mem_before_sync_mib","mem_before_opt_mib","peak_before_opt_mib",
                    "mem_after_opt_mib","peak_during_opt_mib",
                ])
            for rlist in gathered:
                if rlist:
                    for r in rlist:
                        w.writerow(r)
    dist.barrier()
    dist.destroy_process_group()


def run_one(mode, world_size, port):
    label = f"FSDP {mode} world_size={world_size}"
    print(f"\n── {label} (port={port}) ──")
    mp.spawn(_worker, args=(world_size, mode, str(port)),
             nprocs=world_size, join=True)


def summarize():
    groups: dict[tuple[str, int], dict[str, list[float]]] = {}
    with OUT.open() as f:
        for r in csv.DictReader(f):
            key = (r["mode"], int(r["world_size"]))
            if key not in groups:
                groups[key] = {k: [] for k in [
                    "total_s","optimizer_s","forward_s","backward_s",
                    "gradient_sync_s","peak_before_opt_mib","mem_before_opt_mib"]}
            for k in groups[key]:
                groups[key][k].append(float(r[k]))

    print("\n" + "=" * 90)
    hdr = f"{'':<22} {'2-GPU fp32':>14} {'2-GPU bf16':>14} {'4-GPU fp32':>14} {'4-GPU bf16':>14}"
    print(hdr)
    print("-" * 90)
    keys = [
        ("peak_before_opt_mib", "Peak before opt (MiB)"),
        ("mem_before_opt_mib", "Mem before opt (MiB)"),
        ("total_s", "Total step (s)"),
        ("optimizer_s", "Optimizer step (s)"),
        ("forward_s", "Forward (s)"),
        ("backward_s", "Backward (s)"),
        ("gradient_sync_s", "Gradient sync (s)"),
    ]
    for k, label in keys:
        vals = []
        for mode in ["fsdp_fp32","fsdp_bf16"]:
            for ws in [2, 4]:
                if (mode, ws) in groups:
                    vals.append(statistics.mean(groups[(mode, ws)][k]))
                else:
                    vals.append(float("nan"))
        print(f"{label:<22} {vals[0]:>14.1f} {vals[1]:>14.1f} {vals[2]:>14.1f} {vals[3]:>14.1f}")
    print("=" * 90)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if OUT.exists():
        OUT.unlink()
    for mode in ["fsdp_fp32", "fsdp_bf16"]:
        for ws in [2, 4]:
            run_one(mode, ws, 29550 + ws * 10 + (1 if "bf16" in mode else 0))
    summarize()
