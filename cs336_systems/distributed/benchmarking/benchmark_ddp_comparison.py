"""
Quick head-to-head: NaiveDDP vs FlattenDDP gradient sync.
Runs both with batch=4, xl model, 2 GPUs, prints comparison.
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
from cs336_systems.distributed.flatten_ddp import FlattenDDP

WARMUP = 5
STEPS = 10
VOCAB = 10_000
CTX = 512
SEED = 0
XL = {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}
OUT = Path("artifacts/ddp_comparison.csv")


def _sync(d: torch.device) -> None:
    if d.type == "cuda":
        torch.cuda.synchronize(d)


def _step(model, opt, x, y, dev, do_sync):
    opt.zero_grad(set_to_none=True)
    _sync(dev); t0 = timeit.default_timer()
    logits = model(x)
    _sync(dev); fwd = timeit.default_timer() - t0
    _sync(dev); t0 = timeit.default_timer()
    loss = cross_entropy(logits, y)
    _sync(dev); loss_t = timeit.default_timer() - t0
    _sync(dev); t0 = timeit.default_timer()
    loss.backward()
    _sync(dev); bwd = timeit.default_timer() - t0
    if do_sync:
        _sync(dev); t0 = timeit.default_timer()
        model.finish_gradient_synchronization()
        _sync(dev); sync_t = timeit.default_timer() - t0
    else:
        sync_t = 0.0
    _sync(dev); t0 = timeit.default_timer()
    opt.step()
    _sync(dev); opt_t = timeit.default_timer() - t0
    return fwd, loss_t, bwd, sync_t, opt_t


def _worker(rank, world_size, mode, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = BasicsTransformerLM(vocab_size=VOCAB, context_length=CTX, **XL).to(dev)
    model.train()
    wrapper = FlattenDDP(model) if mode == "flatten" else NaiveDDP(model)
    opt = AdamW(wrapper.parameters())
    x = torch.randint(0, VOCAB, (2, CTX), device=dev)
    y = torch.randint(0, VOCAB, (2, CTX), device=dev)

    rows = []
    for _ in range(WARMUP):
        _step(wrapper, opt, x, y, dev, do_sync=True)
    for i in range(STEPS):
        fwd, loss_t, bwd, sync_t, opt_t = _step(wrapper, opt, x, y, dev, do_sync=True)
        total = fwd + loss_t + bwd + sync_t + opt_t
        rows.append((mode, 4, i, rank, fwd, loss_t, bwd, sync_t, opt_t, total))

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        write_header = not OUT.exists()
        with OUT.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["mode","batch_size","step","rank","forward_s","loss_s","backward_s","gradient_sync_s","optimizer_s","total_s"])
            for rlist in gathered:
                if rlist:
                    for r in rlist:
                        w.writerow(r)
    dist.barrier()
    dist.destroy_process_group()


def run_one(mode, port):
    label = "FlattenDDP" if mode == "flatten" else "NaiveDDP"
    print(f"\n── {label} (port={port}) ──")
    mp.spawn(_worker, args=(2, mode, str(port)), nprocs=2, join=True)


def summarize():
    groups: dict[str, dict[str, list[float]]] = {}
    with OUT.open() as f:
        for r in csv.DictReader(f):
            m = r["mode"]
            if m not in groups:
                groups[m] = {"fwd": [], "bwd": [], "sync": [], "opt": [], "total": []}
            groups[m]["fwd"].append(float(r["forward_s"]))
            groups[m]["bwd"].append(float(r["backward_s"]))
            groups[m]["sync"].append(float(r["gradient_sync_s"]))
            groups[m]["opt"].append(float(r["optimizer_s"]))
            groups[m]["total"].append(float(r["total_s"]))
    print("\n" + "=" * 75)
    print(f"{'Segment':<16} {'NaiveDDP (291×)':>20} {'FlattenDDP (1×)':>20} {'Δ':>10}")
    print("-" * 75)
    for seg, label in [("fwd","Forward"), ("bwd","Backward"), ("sync","Gradient sync"), ("opt","Optimizer"), ("total","TOTAL")]:
        n_mean = statistics.mean(groups["naive"][seg])
        f_mean = statistics.mean(groups["flatten"][seg])
        diff = f_mean - n_mean
        print(f"{label:<16} {n_mean:>15.4f}s  {f_mean:>15.4f}s  {diff:>+9.4f}s")
    print("=" * 75)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if OUT.exists():
        OUT.unlink()
    run_one("naive", "29510")
    run_one("flatten", "29511")
    summarize()
