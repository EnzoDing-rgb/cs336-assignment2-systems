"""
Optimizer State Sharding benchmark: baseline (AdamW) vs sharded (ShardedOptimizer).

xl model, 2 GPUs, batch=4. Measures peak memory and step timing.

Output: artifacts/sharded_optimizer_benchmark.csv
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
from cs336_systems.distributed.naive_ddp import NaiveDDP
from cs336_systems.distributed.sharded_optimizer import ShardedOptimizer

WARMUP = 5
STEPS  = 10
VOCAB  = 10_000
CTX    = 512
SEED   = 0
XL     = {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
OUT    = Path("artifacts/sharded_optimizer_benchmark.csv")


# ── Helpers ────────────────────────────────────────────────────────────────
def _sync(d: torch.device) -> None:
    if d.type == "cuda":
        torch.cuda.synchronize(d)


def _mem_mib(d: torch.device) -> float:
    """Current allocated GPU memory in MiB."""
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


# ── Worker ──────────────────────────────────────────────────────────────────
def _worker(rank, world_size, mode, port):
    """mode: 'baseline' (AdamW) or 'sharded' (ShardedOptimizer)."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # ── Model init ──
    _reset_peak(dev)
    model = BasicsTransformerLM(vocab_size=VOCAB, context_length=CTX, **XL).to(dev)
    model.train()
    ddp = NaiveDDP(model)
    mem_after_init = _mem_mib(dev)
    peak_after_init = _peak_mib_since_reset(dev)

    # ── Optimizer ──
    _reset_peak(dev)
    if mode == "sharded":
        opt = ShardedOptimizer(ddp.parameters(), AdamW, lr=1e-3)
    else:
        opt = AdamW(ddp.parameters(), lr=1e-3)
    mem_after_opt_ctor = _mem_mib(dev)
    peak_after_opt_ctor = _peak_mib_since_reset(dev)

    x = torch.randint(0, VOCAB, (2, CTX), device=dev)
    y = torch.randint(0, VOCAB, (2, CTX), device=dev)

    # ── Warmup ──
    for _ in range(WARMUP):
        opt.zero_grad(set_to_none=True)
        loss = cross_entropy(ddp(x), y)
        loss.backward()
        ddp.finish_gradient_synchronization()
        opt.step()
        _sync(dev)

    # ── Measurement ──
    rows = []
    for step_idx in range(STEPS):
        _reset_peak(dev)

        # forward
        opt.zero_grad(set_to_none=True)
        _sync(dev); t0 = timeit.default_timer()
        logits = ddp(x)
        _sync(dev); fwd = timeit.default_timer() - t0

        # loss
        _sync(dev); t0 = timeit.default_timer()
        loss = cross_entropy(logits, y)
        _sync(dev); loss_t = timeit.default_timer() - t0

        # backward
        _sync(dev); t0 = timeit.default_timer()
        loss.backward()
        _sync(dev); bwd = timeit.default_timer() - t0

        mem_before_sync = _mem_mib(dev)

        # gradient sync
        _sync(dev); t0 = timeit.default_timer()
        ddp.finish_gradient_synchronization()
        _sync(dev); sync_t = timeit.default_timer() - t0

        mem_before_opt = _mem_mib(dev)
        peak_before_opt = _peak_mib_since_reset(dev)

        # optimizer step
        _reset_peak(dev)
        _sync(dev); t0 = timeit.default_timer()
        opt.step()
        _sync(dev); opt_t = timeit.default_timer() - t0

        mem_after_opt = _mem_mib(dev)
        peak_during_opt = _peak_mib_since_reset(dev)

        total = fwd + loss_t + bwd + sync_t + opt_t
        rows.append((mode, 4, step_idx, rank,
                     fwd, loss_t, bwd, sync_t, opt_t, total,
                     mem_after_init, peak_after_init,
                     mem_after_opt_ctor, peak_after_opt_ctor,
                     mem_before_sync, mem_before_opt, peak_before_opt,
                     mem_after_opt, peak_during_opt))

        if rank == 0:
            print(f"  step {step_idx+1:>2}/{STEPS}  fwd={fwd:.3f}s  bwd={bwd:.3f}s  "
                  f"sync={sync_t:.3f}s  opt={opt_t:.3f}s  total={total:.3f}s  "
                  f"mem_before_opt={mem_before_opt:.0f}MiB  mem_after_opt={mem_after_opt:.0f}MiB")

    # Gather
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUT.exists()
        with OUT.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow([
                    "mode","batch_size","step","rank",
                    "forward_s","loss_s","backward_s","gradient_sync_s","optimizer_s","total_s",
                    "mem_after_init_mib","peak_after_init_mib",
                    "mem_after_opt_ctor_mib","peak_after_opt_ctor_mib",
                    "mem_before_sync_mib","mem_before_opt_mib","peak_before_opt_mib",
                    "mem_after_opt_mib","peak_during_opt_mib",
                ])
            for rlist in gathered:
                if rlist:
                    for r in rlist:
                        w.writerow(r)
    dist.barrier()
    dist.destroy_process_group()


def run_one(mode, port):
    label = "ShardedOptimizer" if mode == "sharded" else "AdamW (baseline)"
    print(f"\n── {label} (port={port}) ──")
    mp.spawn(_worker, args=(2, mode, str(port)), nprocs=2, join=True)


def summarize():
    groups: dict[str, dict[str, list[float]]] = {}
    with OUT.open() as f:
        for r in csv.DictReader(f):
            m = r["mode"]
            if m not in groups:
                groups[m] = {k: [] for k in ["total_s","optimizer_s",
                    "mem_before_opt_mib","mem_after_opt_mib","peak_before_opt_mib"]}
            for k in groups[m]:
                groups[m][k].append(float(r[k]))

    def mn(m, k):
        return statistics.mean(groups[m][k])

    print("\n" + "=" * 72)
    print(f"{'':<28} {'AdamW':>16} {'Sharded':>16} {'Δ':>10}")
    print("-" * 72)
    print(f"{'Peak before opt (MiB)':<28} {mn('baseline','peak_before_opt_mib'):>16.0f} "
          f"{mn('sharded','peak_before_opt_mib'):>16.0f} "
          f"{mn('sharded','peak_before_opt_mib')-mn('baseline','peak_before_opt_mib'):>+10.0f}")
    print(f"{'Mem before opt (MiB)':<28} {mn('baseline','mem_before_opt_mib'):>16.0f} "
          f"{mn('sharded','mem_before_opt_mib'):>16.0f} "
          f"{mn('sharded','mem_before_opt_mib')-mn('baseline','mem_before_opt_mib'):>+10.0f}")
    print(f"{'Mem after opt (MiB)':<28} {mn('baseline','mem_after_opt_mib'):>16.0f} "
          f"{mn('sharded','mem_after_opt_mib'):>16.0f} "
          f"{mn('sharded','mem_after_opt_mib')-mn('baseline','mem_after_opt_mib'):>+10.0f}")
    print(f"{'Optimizer step (s)':<28} {mn('baseline','optimizer_s'):>16.4f} "
          f"{mn('sharded','optimizer_s'):>16.4f} "
          f"{mn('sharded','optimizer_s')-mn('baseline','optimizer_s'):>+10.4f}")
    print(f"{'Total step (s)':<28} {mn('baseline','total_s'):>16.4f} "
          f"{mn('sharded','total_s'):>16.4f} "
          f"{mn('sharded','total_s')-mn('baseline','total_s'):>+10.4f}")
    print("=" * 72)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if OUT.exists():
        OUT.unlink()
    run_one("baseline", "29540")
    run_one("sharded", "29541")
    summarize()
