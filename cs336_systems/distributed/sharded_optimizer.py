from __future__ import annotations

# ---------------------------------------------------------------------------
# 这份文件在干什么（先读这段，再往下看函数）
# ---------------------------------------------------------------------------
# PyTorch 里 torch.optim.AdamW 本身就会：
#   - 记住「要优化哪些参数」
#   - 为每个参数存 OPT 状态（Adam 的 m/v 等）
#   - step() 时用 .grad 更新这些参数
#
# 我们要的 ShardedOptimizer 是「外壳」：
#   - 对外仍像一个 Optimizer（测试调用 zero_grad / step）
#   - 对内再包一个「真正的」AdamW，但只让它管大约 1/world_size 的参数
#   - step 之后用 broadcast，把各卡更新过的权重对齐
#
# 参数是怎么进到这个类里的？（测试里你看不到 add_param_group 的直接调用）
#   test 写的是：
#       get_sharded_optimizer(sharded_model.parameters(), AdamW, lr=0.1, ...)
#   → 最终会构造 ShardedOptimizer(params=..., optimizer_cls=AdamW, lr=0.1, ...)
#   → __init__ 里调用 super().__init__(params, defaults={})
#   → Optimizer 基类构造函数会遍历 params，并对每个 param group
#        调用「你子类上的」add_param_group(...)
#   所以 add_param_group 是「间接」被基类构造流程叫起来的
# ---------------------------------------------------------------------------

from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:
        """
        params
            「全部要优化的权重」
            常见来源：model.parameters() 这种迭代器，里面每个元素是 nn.Parameter。
            也可以是已经分好的 param group 列表（见下面 add_param_group 对 group 形状的说明）

        optimizer_cls
            底层优化器的「类」，例如 torch.optim.AdamW（注意传的是类本身，不是 AdamW() 实例）

        **kwargs（存进 self.optimizer_kwargs）
            传给底层 AdamW 的超参，例如 lr=0.1, betas=(0.9,0.999), weight_decay=0.1, eps=1e-8。
            测试里 get_sharded_optimizer(..., lr=0.1, ...) 多出来的关键字都会进这里。
        """
        if not dist.is_initialized():
            raise RuntimeError("ShardedOptimizer 需要先 dist.init_process_group(...)")

        # 例: world_size=2 时，某个进程上 self.rank 可能是 0，另一个进程上是 1
        self.rank = dist.get_rank()
        # 例: self.world_size == 2
        self.world_size = dist.get_world_size()

        # 例: self.optimizer_cls is torch.optim.AdamW  （还没实例化）
        self.optimizer_cls = optimizer_cls
        # 例: self.optimizer_kwargs == {"lr": 0.1, "betas": (0.9, 0.999), "weight_decay": 0.1, "eps": 1e-8}
        self.optimizer_kwargs = kwargs

        # 真正干活的 AdamW；例: 稍后变成 AdamW(params=[W0, W2, ...], lr=0.1, ...)
        # 一开始是 None；第一次 add_param_group 时再创建。
        self._local_optimizer: Optimizer | None = None

        # 全局登记表。例（world_size=2、4 个参数时）最终可能长这样：
        #   [
        #     (Parameter(shape=[10,10]), 0),  # 第 0 个参数，归 rank0 更新
        #     (Parameter(shape=[50,10]), 1),  # 第 1 个参数，归 rank1 更新
        #     (Parameter(shape=[10,50]), 0),
        #     (Parameter(shape=[10]),    1),
        #   ]
        # step 末尾：对每一项 (p, owner) 做 broadcast(p.data, src=owner)
        self._param_owners: list[tuple[torch.nn.Parameter, int]] = []

        # ------------------------------------------------------------------
        # 为什么还要 super().__init__(params, defaults={})？
        # ------------------------------------------------------------------
        # Optimizer 基类会维护：
        #   - self.param_groups：外壳也记得「全体参数」（zero_grad 才能清掉所有 .grad）
        #   - 一些 Optimizer 约定状态
        #
        # defaults={}：基类要求的「默认超参字典」。
        #   我们真正的 lr/betas 已经放进 _local_optimizer，所以这里给空字典即可。
        #
        # 调用 super().__init__ 的过程中，基类会（间接）调用本类的 add_param_group，
        # 把 params 装进来并完成分片。所以「完整参数进入优化器」发生在这一行触发的链路里。
        # ------------------------------------------------------------------
        super().__init__(params, defaults={})

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        把一组参数登记进优化器，并完成本 rank 的分片。

        param_group 长什么样？
            一个普通 Python dict，至少有键 "params"。
            最小例子：
                {"params": [weight1, weight2, bias1, ...]}
            若不同层要用不同 lr，还可能带其它键：
                {"params": [...], "lr": 0.01, "weight_decay": 0.0, ...}

        谁调用我？
            1) 最常见：__init__ → super().__init__ → 基类内部调用 add_param_group
               （对应 test 里一次传入 model.parameters()）
            2) 训练中途也可能：opt.add_param_group({...})（讲义要求要支持）

        为什么函数里还要再调 Optimizer.add_param_group(self, group)？
            那是在调用「父类实现」：让外壳 Optimizer 正式登记这组「完整参数」。
            子类 add_param_group 里 = 自己的分片逻辑 + 父类的登记逻辑
        """
        # 浅拷贝。例:
        #   param_group == {"params": <generator or list>, "lr": 0.1}
        #   group       == {"params": 同上的拷贝引用结构, "lr": 0.1}
        group = {k: v for k, v in param_group.items()}

        # 取出参数列表；统一变成 list。
        # 例: params 可能先是 generator；list 之后变成
        #   [Parameter(10x10), Parameter(50x10), Parameter(10x50), Parameter(10)]
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]  # 极端情况：只传进来单个张量
        else:
            params = list(params)
        group["params"] = params  # 例: group["params"] 现在一定是 list[...]

        # (A) 外壳登记完整名单。之后大约有:
        #   self.param_groups == [{"params": [P0,P1,P2,P3], "lr": 0.1, ...}]
        Optimizer.add_param_group(self, group)

        # (B) 分片。下面用「4 个参数、world_size=2、当前是 rank0」当贯穿例子。
        local_params: list[torch.nn.Parameter] = []  # 例: 循环结束后 → [P0, P2]
        seen_in_group: set[int] = set()  # 例: 循环中逐渐变成 {id(P0), id(P1), ...}

        for i, p in enumerate(params):
            # 第 0 轮例: i == 0, p is P0  (某个 nn.Parameter，如 shape [10,10])
            # 第 1 轮例: i == 1, p is P1
            # 第 2 轮例: i == 2, p is P2
            # 第 3 轮例: i == 3, p is P3

            pid = id(p)  # 例: 某个很大的整数，如 140392881234560（对象身份，不是下标）
            if pid in seen_in_group:
                # tied weights：同一个 Parameter 又出现一次 → 跳过，避免重复分片/重复广播
                continue
            seen_in_group.add(pid)  # 例: 第一轮后 seen_in_group == {id(P0)}

            # owner = 谁负责更新这个 p、谁保存它的 Adam 状态
            # 约定: owner = i % world_size
            # 例 world_size=2:
            #   i=0 → owner=0
            #   i=1 → owner=1
            #   i=2 → owner=0
            #   i=3 → owner=1
            owner = i % self.world_size

            # 例: append 四次后 self._param_owners ==
            #   [(P0, 0), (P1, 1), (P2, 0), (P3, 1)]
            self._param_owners.append((p, owner))

            # 只有 owner 是「我」才放入本地 AdamW
            # 例: 当前 self.rank == 0 时，i=0、i=2 会进来 → local_params == [P0, P2]
            # 例: 当前 self.rank == 1 时，会得到 local_params == [P1, P3]
            if owner == self.rank:
                local_params.append(p)

        # 给底层 AdamW 的 group：去掉完整 params，换成本分片。
        # 例 rank0:
        #   local_group == {"lr": 0.1, "params": [P0, P2]}   # 若原 group 里有 lr
        # 若原 group 只有 params 键:
        #   local_group == {"params": [P0, P2]}
        local_group = {k: v for k, v in group.items() if k != "params"}
        local_group["params"] = local_params

        # (C) 创建/扩充本地 AdamW。OPT 状态（m/v）只会为 local_params 里那些张量分配。
        if self._local_optimizer is None:
            # 例: 等价于 AdamW([{"params": [P0, P2], ...}], lr=0.1, betas=..., ...)
            self._local_optimizer = self.optimizer_cls(
                [local_group],
                **self.optimizer_kwargs,
            )
        else:
            # 中途又 add_param_group 时走这里
            self._local_optimizer.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        """
        一步更新：
          1) 底层 AdamW 只用本分片的 .grad，只改本分片的 .data，只动本分片 OPT 状态
          2) 全员参与 broadcast：每个参数以它的 owner 为 src，把新权重同步到所有 rank
        """
        if self._local_optimizer is None:
            raise RuntimeError("本地 optimizer 尚未创建（没有 param group？）")

        # 例 rank0: 只更新 P0、P2 的 .data；P1、P3 在本卡上暂时仍是旧值
        loss = self._local_optimizer.step(closure, **kwargs)

        # 例: self._param_owners == [(P0,0),(P1,1),(P2,0),(P3,1)]
        # 第一轮: p is P0, owner==0 → 全体执行 broadcast(P0.data, src=0)
        # 第二轮: p is P1, owner==1 → 全体执行 broadcast(P1.data, src=1)
        # …
        # 结束后每张卡上的 P0..P3 都与对应 owner 上的新值一致
        for p, owner in self._param_owners:
            dist.broadcast(p.data, src=owner)

        return loss
