from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class OverlapDDP(nn.Module):

    def __init__(self, module: nn.Module) -> None:
        """
        module: 被包装的真正模型（例如 ToyModel 实例）。
        """
        super().__init__()

        self.module: nn.Module = module

        # 异步 all_reduce 返回的「收据」列表。
        # dist.Work：一次尚未完成的集合通信的句柄；.wait() 等到结束。
        self._handles: list[dist.Work] = []

        self._broadcast_parameters()
        self._register_grad_hooks() #构造的时候调用

    def _broadcast_parameters(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError("OverlapDDP 需要先 dist.init_process_group(...)")

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def _register_grad_hooks(self) -> None:
        """
        给每个可训练参数登记 post-accumulate 钩子。
        登记 ≠ 立刻执行；真正调用发生在之后的 loss.backward() 里。
        """
        seen: set[int] = set()

        for param in self.module.parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)

            if not param.requires_grad:
                continue

            param.register_post_accumulate_grad_hook(self._grad_hook)

    def _grad_hook(self, param: torch.Tensor) -> None:
        """
        autograd 在 backward 中途调用：对该 param.grad 做异步 all_reduce。
        """
        if param.grad is None:
            return

        # Gloo（测试用的 CPU backend）不支持 ReduceOp.AVG，所以用 SUM，
        # 平均放到 finish 里 wait 之后再 div_(world_size)。
        handle: dist.Work = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        self._handles.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        for handle in self._handles:
            handle.wait()
        self._handles.clear()

        # wait 之后每个参与通信的 .grad 里是「各卡之和」，再除以 world_size 得到平均。
        world_size = dist.get_world_size()
        seen: set[int] = set()
        for param in self.module.parameters():
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if param.grad is not None:
                param.grad.div_(world_size)
