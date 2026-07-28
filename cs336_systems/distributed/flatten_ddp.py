from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class FlattenDDP(nn.Module):
    """
    和 NaiveDDP 一样：broadcast + forward + finish 里同步梯度。
    唯一差别：finish 里用 flatten（拼成一条）只 all_reduce 一次，再拆回。
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self._broadcast_parameters()

    def _broadcast_parameters(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "FlattenDDP 需要先 dist.init_process_group(...)。"
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

        # 收集「每个参数自己的 .grad」。
        # seen：同一个 Parameter 因 weight tying 可能出现两次，只收一次，避免重复打包。
        seen: set[int] = set()
        grads: list[torch.Tensor] = []

        for param in self.module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)

            # 没有梯度的参数（如 requires_grad=False）跳过。
            if param.grad is None:
                continue

            # 注意：这里放进列表的是「参数身上那块 grad 张量」本身，不是拷贝。
            # 后面 copy_ 会写回这块内存，也就写回了 param.grad。
            grads.append(param.grad)

        if not grads:
            return

        # ----- flatten：把多块 grad 在内存里首尾相接，变成一条长向量 -----
        # 例：grads = [[1,2], [3], [4,5,6]]  →  flat = [1,2,3,4,5,6]
        flat = torch._utils._flatten_dense_tensors(grads)

        # 只通信这一条。各卡对应位置求和，结果写回每张卡自己的 flat。
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)

        # 求和 → 平均（和 NaiveDDP 里对每个 grad 做 div_ 一样）。
        flat.div_(world_size)

        # ----- unflatten：按原来每块的形状/长度，把长向量拆回多块 -----
        # synced 是一个「新张量」的列表，长度和 grads 相同，第 i 块对应 grads[i] 的形状。
        # 例：flat 平均后是 [5.5, 11, 16.5, 22, 27.5, 33]
        #     → synced[0]=[5.5,11], synced[1]=[16.5], synced[2]=[22,27.5,33]
        #
        # 为什么不直接改 grads？
        #   unflatten 返回的是新分配出来的张量，还没挂回 param.grad。
        #   所以下一步必须把数值写回原来的 grads[i]（也就是各 param.grad）。
        synced = torch._utils._unflatten_dense_tensors(flat, grads)

        # zip(grads, synced)：两两配对
        #   第 0 轮：g = grads[0]（原来的 param.grad），s = synced[0]（拆出来的新块）
        #   第 1 轮：g = grads[1]，s = synced[1]
        #   …
        # g.copy_(s)：把 s 的数值拷进 g 的内存里（形状必须一样）。
        # 不用 g = s：那样只改了局部变量名，param.grad 还是旧数据。
        for g, s in zip(grads, synced):
            g.copy_(s)
