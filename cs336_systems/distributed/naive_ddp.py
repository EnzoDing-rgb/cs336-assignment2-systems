from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    """
    朴素数据并行包装器。

    整步长什么样（2 卡、batch 切成两半）：
      GPU0 吃样本 [x0,x1]          GPU1 吃样本 [x2,x3]
      前向 → 反向                  前向 → 反向
      得到局部梯度 g0              得到局部梯度 g1
      finish: all_reduce 每个参数  finish: 同步做同样的事
      → 每卡 .grad 都变成 (g0+g1)/2
      然后各自 optimizer.step()    （两边数值相同，权重保持同步）

    和 FlattenDDP 的唯一差别：这里对 291 个 param.grad 各调一次 all_reduce；
    FlattenDDP 先拼成一条再只调一次。
    """

    def __init__(self, module: nn.Module) -> None:
        # 例：module = BasicsTransformerLM(...).cuda(rank)
        #     此刻每张卡上的权重是各自随机初始化的，数值互不相同。
        super().__init__()
        self.module = module  # 真正干活的模型；NaiveDDP 自己几乎不存参数
        self._broadcast_parameters()  # 立刻把 rank0 的权重广播到所有卡

    def _broadcast_parameters(self) -> None:
        # 训练开始前必须保证：所有卡上的 W 完全一样。
        # 否则后面 all_reduce 梯度、各自 step，权重会越走越歪。
        if not dist.is_initialized():
            raise RuntimeError(
                "NaiveDDP 需要先 dist.init_process_group(...)。"
                "（测试里 _setup_process_group 会先做这件事。）"
            )

        # 例：2 卡，某个 Linear.weight 形状 (4, 2)
        #   广播前:
        #     rank0: [[1,2],[3,4],[5,6],[7,8]]
        #     rank1: [[9,9],[9,9],[9,9],[9,9]]   ← 随机 init，和 rank0 不同
        #   dist.broadcast(param.data, src=0) 之后:
        #     rank0: [[1,2],[3,4],[5,6],[7,8]]
        #     rank1: [[1,2],[3,4],[5,6],[7,8]]   ← 被覆盖成 rank0 的值
        # param.data：参数本体的存储（不是 .grad）；in-place 改这块内存。
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # buffer 不是 Parameter，但前向也会用到（如 BatchNorm 的 running_mean）。
        # 例：某 buffer = tensor([0.1, 0.2]) 在 rank0，rank1 上可能是 [0.0, 0.0]
        #     broadcast 后两边都是 [0.1, 0.2]。
        # 本作业 Transformer 几乎没有需要同步的 buffer，但写上更稳妥。
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def forward(self, *args, **kwargs):
        # 例：logits = ddp_model(x)  等价于  logits = ddp_model.module(x)
        # NaiveDDP 不改计算图，只是把调用转给里面的 module。
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        # 调用时机（训练循环里）：
        #   loss.backward()                         # 每卡算出自己那份局部 .grad
        #   ddp.finish_gradient_synchronization()   # ← 本函数：变成全局平均 .grad
        #   optimizer.step()                        # 用平均梯度更新本卡完整权重
        if not dist.is_initialized():
            raise RuntimeError("finish_gradient_synchronization 需要已初始化的进程组")

        # 例：2 卡 → world_size = 2；4 卡 → 4。
        # 后面用它把「求和」变成「平均」：sum / N。
        world_size = dist.get_world_size()

        # 具体例子：tests/common.py 里的 ToyModelWithTiedWeights
        #   self.fc2 = nn.Linear(10, 50, bias=False)   # 权重 W，形状 (50, 10)
        #   self.fc4 = nn.Linear(10, 50, bias=False)
        #   self.fc4.weight = self.fc2.weight          # fc4 不新建 W，和 fc2 共用同一块
        #
        #   因此 Python 里只有 1 个 Parameter 对象，比如 id(W) = 140234567890。
        #   但 for param in model.parameters() 会走到它两次：
        #     第 1 次：param is fc2.weight  → id = 140234567890
        #     … 中间别的层 …
        #     第 2 次：param is fc4.weight → id 还是 140234567890（同一块显存）
        #
        #   backward 后：fc2、fc4 两路的梯度都写进 W.grad（autograd 自动累加），本来就只有 1 份。
        #   2 卡、局部梯度 rank0=[3,5]、rank1=[7,1] 时：
        #     第 1 次 all_reduce+div → 两卡 W.grad 都变成 [5, 3]  ✓ 全局平均
        #     若没有 seen、第 2 次又对同一块 W.grad 再 all_reduce+div：
        #       两卡已是 [5,3]，再 SUM → [10,6]，再 /2 → 仍是 [5,3]（数值碰巧不变）
        #       但白跑 1 次 NCCL；更糟的是你会误以为有「两个参数」各同步了一次。
        #   seen 的作用：同一个 id 只进循环体一次，共享权重只 all_reduce 一次。
        seen: set[int] = set()

        # xl 模型大约 291 个 Parameter → 本循环最多触发 291 次 all_reduce。
        for param in self.module.parameters():
            param_id = id(param)  # 对象身份，不是「第几个参数」的下标
            if param_id in seen:
                continue
            seen.add(param_id)

            # 例：某层 requires_grad=False，或这步没参与 loss → param.grad is None
            #     没有东西可同步，跳过，否则 all_reduce(None) 会炸。
            if param.grad is None:
                continue

            # ----- 核心：对「这一块」梯度做一次 all_reduce -----
            # 例：2 卡，某个 bias.grad 长度 3
            #   调用前（各卡只有局部梯度）:
            #     rank0: param.grad = [1.0, 2.0, 3.0]     ← 只看过自己那半 batch
            #     rank1: param.grad = [5.0, 6.0, 7.0]
            #   dist.all_reduce(..., SUM) 之后（就地改写，每卡都变成和）:
            #     rank0: param.grad = [6.0, 8.0, 10.0]
            #     rank1: param.grad = [6.0, 8.0, 10.0]
            # 注意：这里一次只传这一块张量。291 个参数 = 291 次独立 NCCL 调用。
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

            # 求和 → 平均（和「全局 batch 上的梯度」同尺度）
            #   接上例，world_size=2:
            #     [6.0, 8.0, 10.0] / 2 → [3.0, 4.0, 5.0]
            #   div_ 是 in-place：直接改 param.grad 这块显存，不新开张量。
            #   之后 optimizer.step() 读到的就是这份平均梯度。
            param.grad.div_(world_size)
