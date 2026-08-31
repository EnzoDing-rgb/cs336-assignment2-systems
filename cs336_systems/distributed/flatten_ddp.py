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
        # 调用时机（训练循环里）：
        #   loss.backward()                         # 每卡算出自己那份局部 .grad
        #   ddp.finish_gradient_synchronization()   # ← 本函数：变成全局平均 .grad
        #   optimizer.step()                        # 用平均梯度更新本卡完整权重
        #
        # 和 NaiveDDP 的目标完全一样，只是手段不同：
        #   NaiveDDP：对 291 个 param.grad 各调一次 all_reduce（291 次 NCCL）
        #   FlattenDDP：先把 291 块 grad 拼成 1 条长向量，只 all_reduce 1 次
        if not dist.is_initialized():
            raise RuntimeError("finish_gradient_synchronization 需要已初始化的进程组")

        # 例：2 卡 → world_size = 2；4 卡 → 4。
        # 后面用它把「求和」变成「平均」：sum / N。
        world_size = dist.get_world_size()

        # weight tying 例：embedding.weight 和 lm_head.weight 是同一个 Parameter 对象
        #   list(model.parameters()) 里会出现两次同一个 id
        #   若把同一块 .grad 放进 grads 两次 → flatten 里会重复打包 → all_reduce 后数值翻倍，错。
        seen: set[int] = set()

        # grads 是什么？
        #   一个「指针列表」：按遍历顺序，依次记下「每个需要同步的 param.grad 张量」。
        #   注意：存的是引用，不是 clone；grads[i] 和某个 param.grad 是同一块显存。
        #
        # 为什么要单独建这个列表？（不能直接遍历 module.parameters() 一路 flatten 吗？）
        #   1. flatten / unflatten API 要的是 list[Tensor]，不是 Module 里的 Parameter 迭代器
        #   2. 要先过滤掉 grad is None 的参数
        #   3. weight tying 要去重（seen）
        #   4. all_reduce 之后 unflatten 需要「原来的形状模板」——就是这份 grads 列表
        #   5. 最后 copy_ 时，要知道写回哪几块内存 → 还是这份 grads
        #
        # 具体例子（假设模型只有 3 个有梯度的参数，2 卡）：
        #   param_A.weight.grad  shape (2,)   rank0=[1.0, 2.0]   rank1=[5.0, 6.0]
        #   param_B.weight.grad  shape (1,)   rank0=[3.0]        rank1=[7.0]
        #   param_C.weight.grad  shape (3,)   rank0=[4.0,5.0,6.0] rank1=[8.0,9.0,10.0]
        #
        #   遍历结束后，rank0 上的 grads 是：
        #     grads[0] → param_A.weight.grad  即 tensor([1.0, 2.0])
        #     grads[1] → param_B.weight.grad  即 tensor([3.0])
        #     grads[2] → param_C.weight.grad  即 tensor([4.0, 5.0, 6.0])
        #   （rank1 上 grads 的结构相同，只是里面的数值不同）
        grads: list[torch.Tensor] = []

        for param in self.module.parameters():
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)

            # 例：某层 requires_grad=False，或这步没参与 loss → param.grad is None
            #     没有东西可同步，跳过。
            if param.grad is None:
                continue

            # grads.append(param.grad) 不是拷贝：
            #   设 param.grad 在显存地址 0xAAA
            #   append 之后 grads[-1] 也指向 0xAAA
            #   改 grads[-1] 就等于改 param.grad（同一块内存）
            grads.append(param.grad)

        # 例：整个模型没有任何 .grad（没 backward 过）→ 无事可做
        if not grads:
            return

        # ----- 第 1 步：flatten —— 把 grads 里多块梯度首尾相接，变成一条长向量 -----
        #
        # 接上例，rank0：
        #   grads = [ [1,2], [3], [4,5,6] ]     ← 3 个独立张量，共 6 个元素
        #   flat  = [1, 2, 3, 4, 5, 6]          ← 1 个张量，numel=6
        #
        # rank1 上 flat 对应位置是 [5, 6, 7, 8, 9, 10]（局部梯度不同，但形状一样）
        #
        # flat 是新分配的一块连续显存；grads 里那 6 个数还在原来的 3 块里，暂时没变。
        flat = torch._utils._flatten_dense_tensors(grads)

        # ----- 第 2 步：all_reduce —— 只通信 flat 这一条，不再 291 次 -----
        #
        # 2 卡 SUM 之后，每张卡上的 flat 都变成对应位置相加：
        #   flat = [1+5, 2+6, 3+7, 4+8, 5+9, 6+10]
        #        = [6, 8, 10, 12, 14, 16]
        #
        # 此时 grads 里的旧局部值 [1,2], [3], [4,5,6] 仍然没动（all_reduce 改的是 flat）。
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)

        # ----- 第 3 步：求和 → 平均（和 NaiveDDP 里 param.grad.div_(world_size) 一样）-----
        #
        #   flat = [6,8,10,12,14,16] / 2 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        #
        # 若用 NaiveDDP，等价于：
        #   param_A.grad = [3.0, 4.0]
        #   param_B.grad = [5.0]
        #   param_C.grad = [6.0, 7.0, 8.0]
        flat.div_(world_size)

        # ----- 第 4 步：unflatten —— 按 grads 里各块的形状，把 flat 拆回多块 -----
        #
        # 形状模板来自 grads（不是随便切的）：
        #   grads[0] 有 2 个元素 → synced[0] shape (2,)
        #   grads[1] 有 1 个元素 → synced[1] shape (1,)
        #   grads[2] 有 3 个元素 → synced[2] shape (3,)
        #
        #   flat = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        #   synced[0] = [3.0, 4.0]
        #   synced[1] = [5.0]
        #   synced[2] = [6.0, 7.0, 8.0]
        #
        # 关键：synced[i] 是新张量，和 grads[i] 不是同一块显存！
        #   grads[0] 此时仍是 [1.0, 2.0]（旧的局部梯度）
        #   synced[0] 已经是 [3.0, 4.0]（平均后的正确值）
        #   param_A.weight.grad 还指着 grads[0] 那块旧内存 → 必须 copy_ 写回去。
        synced = torch._utils._unflatten_dense_tensors(flat, grads)

        # ----- 第 5 步：copy_ —— 把平均结果写回原来的 param.grad -----
        #
        # zip(grads, synced) 按收集时的顺序一一配对：
        #   轮次 0: g = grads[0] = param_A.weight.grad  旧 [1.0, 2.0]
        #           s = synced[0]                         新 [3.0, 4.0]
        #           g.copy_(s) → param_A.weight.grad 变成 [3.0, 4.0]
        #   轮次 1: g = grads[1] = param_B.weight.grad  [3.0] → [5.0]
        #   轮次 2: g = grads[2] = param_C.weight.grad  [4,5,6] → [6,7,8]
        #
        # 为什么不用 g = s？
        #   g = s 只让局部变量 g 指向新张量 s，param.grad 仍指着旧内存，optimizer 读到的是错的。
        #   g.copy_(s) 是 in-place 写入 g 指向的那块显存，也就是 param.grad 本体。
        for g, s in zip(grads, synced):
            g.copy_(s)
