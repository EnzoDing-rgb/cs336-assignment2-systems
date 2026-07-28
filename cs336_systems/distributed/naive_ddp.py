from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._broadcast_parameters()

    def _broadcast_parameters(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "NaiveDDP 需要先 dist.init_process_group(...)。"
                "（测试里 _setup_process_group 会先做这件事。）"
            )

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("finish_gradient_synchronization 需要已初始化的进程组")

        world_size = dist.get_world_size()

        seen: set[int] = set()

        for param in self.module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)

            if param.grad is None:
                continue

            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)
