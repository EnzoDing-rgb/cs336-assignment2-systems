"""
Minimal script for nsys profiling: 1 warmup + 2 measurement steps.
Run via: nsys profile -o output python _nsys_target.py naive|overlap
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.distributed.naive_ddp import NaiveDDP
from cs336_systems.distributed.overlap_ddp import OverlapDDP

XL = {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32}


def _worker(rank, world_size, mode, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model = BasicsTransformerLM(vocab_size=10000, context_length=512, **XL).to(dev)
    model.train()
    wrapper = OverlapDDP(model) if mode == "overlap" else NaiveDDP(model)
    opt = AdamW(wrapper.parameters())
    x = torch.randint(0, 10000, (2, 512), device=dev)
    y = torch.randint(0, 10000, (2, 512), device=dev)

    # 1 warmup
    opt.zero_grad(set_to_none=True)
    loss = cross_entropy(wrapper(x), y)
    loss.backward()
    wrapper.finish_gradient_synchronization()
    opt.step()
    torch.cuda.synchronize(dev)

    # 2 measured steps
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        logits = wrapper(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        wrapper.finish_gradient_synchronization()
        opt.step()
        torch.cuda.synchronize(dev)
        if rank == 0:
            print(f"step {step+1}/2 done", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "naive"
    mp.set_start_method("spawn", force=True)
    mp.spawn(_worker, args=(2, mode, "29530"), nprocs=2, join=True)
