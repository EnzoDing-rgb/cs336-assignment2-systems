# cs336_systems/distributed/fsdp.py
# ---------------------------------------------------------------------------
# 这份文件在干什么（先读这段）
# ---------------------------------------------------------------------------
# OverlapDDP：每卡常驻完整权重；backward 里对完整梯度做 async all-reduce。
#
# FSDP 再进一步：Linear / Embedding 的 weight 平时只存 1/world_size 分片。
#   - forward / backward 算 matmul 前：all-gather 临时拼出完整权重
#   - 算完：丢掉完整视图，只留分片
#   - 梯度就绪：reduce-scatter，每卡只留自己那截梯度，再 optimizer.step
#
# 贯穿数字例子（与 tests/test_fsdp.py 的 Toy 一致）：
#   world_size=2, rank=0 或 1
#   linear1.weight 完整形状 [128, 64]
#     rank0 分片 [0:64, :]  → shape [64, 64]
#     rank1 分片 [64:128, :] → shape [64, 64]
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

# 作业模型用的是 cs336_basics 的层，不是 torch.nn.Linear
from cs336_basics.model import Embedding, Linear  # RMSNorm 不切分


@dataclass
class ShardMeta:
    """
    某一个「被 FSDP 切开的模块」的元数据。

    例: mod = linear1, world_size=2, rank=0 时大致为:
      full_shape = torch.Size([128, 64])
      shard_dim  = 0
      shard_size = 64
      index      = 1          # 在 _sharded_modules 里排第 2（0-based）
      full       = None       # 尚未 all-gather；或 Tensor([128,64])
      all_gather_work = None  # 或某个 dist.Work（异步 all-gather 的收据）
      is_full    = False      # True 表示 mod.weight.data 当前已是完整 [128,64]
      saved_master = None     # install 完整权重前备份的 fp32 分片 [64,64]
    """

    full_shape: torch.Size
    shard_dim: int
    shard_size: int
    index: int
    full: Tensor | None = None
    all_gather_work: dist.Work | None = None
    is_full: bool = False
    saved_master: Tensor | None = None


class FSDP(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        compute_dtype: torch.dtype | None = None,
    ) -> None:
        """
        module
            被包装的真正模型。例: ToyFSDPModel()，里面有 embedding / norm / linear…

        compute_dtype
            例: torch.float16
            若给定：all-gather 前先把分片 cast 成该 dtype 再通信/计算（省带宽）；
            本地 master 分片与 optimizer 仍用 fp32。
            若为 None：全程 fp32。
        """
        super().__init__()

        # 例: self.module 是 ToyFSDPModel 实例
        self.module: nn.Module = module
        # 例: self.compute_dtype is torch.float16  或  None
        self.compute_dtype: torch.dtype | None = compute_dtype

        # 被切开的模块，按 modules() 遍历到的顺序排列。
        # 例 Toy 最终:
        #   self._sharded_modules == [embedding, linear1, linear2, lm_head]
        #   （norm1 / norm2 不在表里）
        # 有了下标才能做「层 i 的 forward 结束后，预取层 i+2」：
        #   例 i=1 (linear1) → 预取 _sharded_modules[3] (lm_head)
        self._sharded_modules: list[nn.Module] = []

        # 例: self._meta[id(linear1)].shard_size == 64
        self._meta: dict[int, ShardMeta] = {}

        # 例: {id(embedding.weight), id(linear1.weight), id(linear2.weight), id(lm_head.weight)}
        # 用来区分：这些走 reduce-scatter；其余（norm）走 all-reduce
        self._sharded_param_ids: set[int] = set()

        # 异步梯度通信的收据列表。
        # 例 backward 中途可能变成:
        #   [Work(reduce-scatter linear1), Work(all-reduce norm1), Work(...), ...]
        # finish_gradient_synchronization 里对每个 .wait()
        self._grad_works: list[dist.Work] = []

        # 构造时：切开参数 + 登记 hook（登记 ≠ 立刻通信）
        self._shard_parameters()
        self._register_hooks()

    def _shard_parameters(self) -> None:
        """
        把 Linear / Embedding 的 weight 切成本地 1/world_size；RMSNorm 等跳过。

        例 linear1、world_size=2、rank=0:
          切之前: mod.weight.shape == [128, 64]
          切之后: mod.weight.shape == [64, 64]   # 原 [0:64, :]
          optimizer = SGD(fsdp.parameters()) 之后只会为这份 [64,64] 建状态
        """
        if not dist.is_initialized():
            raise RuntimeError("FSDP 需要先 dist.init_process_group(...)")

        self.rank: int = dist.get_rank()  # 例: 0
        self.world_size: int = dist.get_world_size()  # 例: 2

        for mod in self.module.modules():
            # 例: 遇到 RMSNorm → continue（整份留在每张卡上）
            if not isinstance(mod, (Linear, Embedding)):
                continue

            w: Tensor = mod.weight.data
            # 例 linear1: full_shape == torch.Size([128, 64])
            # 例 embedding: full_shape == torch.Size([100, 64])
            full_shape: torch.Size = w.shape
            shard_dim: int = 0  # 始终沿第 0 维切（d_out 或 vocab）

            if full_shape[shard_dim] % self.world_size != 0:
                raise ValueError(
                    f"无法沿 dim={shard_dim} 均分 {full_shape} 到 world_size={self.world_size}"
                )

            # 例: 128 // 2 == 64
            shard_size: int = full_shape[shard_dim] // self.world_size
            # 例 rank0: start=0；rank1: start=64
            start: int = self.rank * shard_size

            # 例 rank0: shard.shape == [64, 64]，内容是原 weight[0:64, :]
            shard: Tensor = w.narrow(shard_dim, start, shard_size).contiguous().clone()

            # 换成新的 Parameter。之后:
            #   list(fsdp.parameters()) 里对应项 shape 就是 [64, 64]
            # 注意：训练过程中我们只改 .data，不换成另一个 Parameter 对象，
            # 这样 optimizer 里握着的引用始终有效。
            mod.weight = nn.Parameter(shard)

            # 例: embedding 第一个被切 → idx=0；linear1 → idx=1；…
            idx: int = len(self._sharded_modules)
            self._sharded_modules.append(mod)
            self._sharded_param_ids.add(id(mod.weight))

            self._meta[id(mod)] = ShardMeta(
                full_shape=full_shape,  # 例: [128, 64]
                shard_dim=shard_dim,  # 例: 0
                shard_size=shard_size,  # 例: 64
                index=idx,  # 例: 1
            )

    def _register_hooks(self) -> None:
        """
        只登记回调；真正执行发生在之后的 forward / loss.backward() 里。

        被切模块挂 4 类（例 linear1）:
          forward_pre_hook              → 算前装上完整权重
          forward_hook                  → 算完丢掉完整权重，并预取 i+2
          full_backward_pre_hook        → 反传该层前再装上完整权重
          post_accumulate_grad_hook     → 梯度就绪后 async reduce-scatter

        未切参数（例 norm1.weight）只挂:
          post_accumulate_grad_hook     → async all-reduce（保持跨卡一致）
        """
        for mod in self._sharded_modules:
            # 用工厂把「当前这个 mod」绑进闭包，避免 for 循环变量坑
            mod.register_forward_pre_hook(self._make_forward_pre_hook(mod))
            mod.register_forward_hook(self._make_forward_post_hook(mod))
            mod.register_full_backward_pre_hook(self._make_backward_pre_hook(mod))
            mod.weight.register_post_accumulate_grad_hook(self._make_grad_hook(mod))

        for param in self.module.parameters():
            if id(param) in self._sharded_param_ids:
                continue
            if not param.requires_grad:
                continue
            # 例: param is norm1.weight，shape == [64]，每卡一份完整副本
            param.register_post_accumulate_grad_hook(self._replicated_grad_hook)

    def _comm_dtype(self, shard: Tensor) -> torch.dtype:
        """
        这次通信/计算该用什么 dtype。

        例: self.compute_dtype is float16 → 返回 torch.float16
        例: self.compute_dtype is None    → 返回 shard.dtype（通常 float32）
        """
        if self.compute_dtype is not None:
            return self.compute_dtype
        return shard.dtype

    def _start_all_gather(
        self,
        mod: nn.Module,
        async_op: bool = True,
        force_dtype: torch.dtype | None = None,
    ) -> dist.Work | None:
        """
        发起一次 all-gather：各卡的分片 → meta.full 里的完整权重。

        例 linear1, world_size=2, async_op=True:
          rank0 投入 shard shape [64, 64]（原行 0:64）
          rank1 投入 shard shape [64, 64]（原行 64:128）
          结束后 meta.full.shape == [128, 64]（可能仍在传输，要看 Work）

        force_dtype
            例: torch.float32 —— Embedding 的 backward 强制用 fp32 完整权重
                （与 test_fsdp 非并行混合精度基线一致）

        调用前要求: meta.is_full == False（当前 mod.weight 必须是分片形态，
        否则投入 all-gather 的不是「一块分片」）。
        """
        meta: ShardMeta = self._meta[id(mod)]
        if meta.is_full:
            raise RuntimeError("all-gather 时 weight 仍是完整形态，应先 _free_full")

        # 已经有一份正在传 / 已传完的 full，且 dtype 对得上 → 复用，别再发一次
        if meta.all_gather_work is not None or meta.full is not None:
            want: torch.dtype = (
                force_dtype
                if force_dtype is not None
                else self._comm_dtype(mod.weight.data)
            )
            # 例: 预取时已经是 fp16 的 [128,64]，forward 也要 fp16 → 直接返回
            if meta.full is not None and meta.full.dtype == want:
                return meta.all_gather_work
            # 例: 预取了 fp16，但 Embedding backward 要 fp32 → 丢掉重来
            if meta.all_gather_work is not None:
                meta.all_gather_work.wait()
            meta.all_gather_work = None
            meta.full = None

        # 例: shard.shape == [64, 64], dtype float32（master）
        shard: Tensor = mod.weight.data
        if force_dtype is not None:
            comm_dtype: torch.dtype = force_dtype  # 例: float32
        else:
            comm_dtype = self._comm_dtype(shard)  # 例: float16

        # 例: fp32 shard → fp16 再通信（compute_dtype 路径）
        shard_comm: Tensor = shard.to(comm_dtype).contiguous()

        # 例: full 先是 empty([128, 64], dtype=float16)
        full: Tensor = shard_comm.new_empty(meta.full_shape)
        work: dist.Work | bool | None = dist.all_gather_into_tensor(
            full,
            shard_comm,
            async_op=async_op,
        )

        # 例: meta.full 现在指向那块 [128, 64] 缓冲区（async 时内容可能还未写完）
        meta.full = full
        if async_op:
            # 例: work 是 dist.Work；调用方以后 .wait() 才能读 full
            assert isinstance(work, dist.Work)
            meta.all_gather_work = work
            return work

        # async_op=False：返回时 full 已就绪；没有 Work 需要等
        meta.all_gather_work = None
        return None

    def _wait_and_install_full(
        self,
        mod: nn.Module,
        force_dtype: torch.dtype | None = None,
    ) -> None:
        """
        确保 mod.weight.data 已经是「完整权重」，供本层计算使用。

        例 linear1、compute_dtype=None:
          进入前: mod.weight.shape == [64, 64]
          离开后: mod.weight.shape == [128, 64]，is_full=True
          且仍是「同一个」nn.Parameter 对象（只换了 .data）

        例 Embedding、compute_dtype=float16、force_dtype=float32（backward）:
          装上的是 fp32 的完整 [100, 64]，不是 fp16
        """
        meta: ShardMeta = self._meta[id(mod)]

        if force_dtype is not None:
            want: torch.dtype = force_dtype
        elif self.compute_dtype is not None:
            want = self.compute_dtype
        else:
            want = torch.float32

        # 已经完整，但 dtype 不对（例: 身上是 fp16，现在要 fp32）→ 先卸掉再重来
        if meta.is_full:
            if mod.weight.data.dtype == want:
                return
            self._free_full(mod)

        # 若预取已经发出：先等传完
        # 例: forward_post(linear1) 预取了 lm_head；进入 lm_head 的 forward_pre 时在这里 wait
        if meta.all_gather_work is not None:
            meta.all_gather_work.wait()
            meta.all_gather_work = None

        # 没有预取，或预取 dtype 不对 → 同步 all-gather 一次
        if meta.full is None or meta.full.dtype != want:
            meta.full = None
            self._start_all_gather(mod, async_op=False, force_dtype=force_dtype)

        assert meta.full is not None

        # 备份当前 fp32 分片。例: saved_master.shape == [64, 64], float32
        # free 时原样写回，避免「完整权重是 fp16 → 再切回 fp32」的精度损失
        # （与 test_fsdp._apply_mixed_precision_hooks 里的 _saved_fp32 同一思想）
        if meta.saved_master is None:
            meta.saved_master = (
                mod.weight.data.detach().to(torch.float32).contiguous().clone()
            )

        # 例: 之后 mod.weight.data is meta.full，shape [128, 64]
        with torch.no_grad():
            mod.weight.data = meta.full
        meta.is_full = True

    def _free_full(self, mod: nn.Module) -> None:
        """
        丢掉临时完整权重，写回本地 fp32 分片。

        例 linear1:
          进入前: mod.weight.shape == [128, 64]，is_full=True
          离开后: mod.weight.shape == [64, 64]，内容 == install 前的 saved_master
        """
        meta: ShardMeta = self._meta[id(mod)]
        if not meta.is_full:
            # 可能只是预取了 meta.full，还没 install 到 mod.weight
            meta.full = None
            meta.all_gather_work = None
            return

        if meta.saved_master is None:
            # 兜底：从完整张量上切回本 rank 那一块（正常混合精度路径很少走到）
            # 例 rank0: full[0:64, :].to(float32)
            full: Tensor = mod.weight.data
            start: int = self.rank * meta.shard_size
            shard: Tensor = (
                full.narrow(meta.shard_dim, start, meta.shard_size)
                .contiguous()
                .to(torch.float32)
                .clone()
            )
        else:
            # 例: 直接拿回备份的 [64, 64] fp32
            shard = meta.saved_master

        with torch.no_grad():
            mod.weight.data = shard
        meta.is_full = False
        meta.full = None
        meta.all_gather_work = None
        meta.saved_master = None

    def _make_forward_pre_hook(self, mod: nn.Module):
        """
        返回给 register_forward_pre_hook 的回调。

        例: 即将执行 linear1(x) 之前调用 → 先保证完整权重已装上。
        """

        def hook(module: nn.Module, inputs: tuple) -> None:
            self._wait_and_install_full(mod)

        return hook

    def _make_forward_post_hook(self, mod: nn.Module):
        """
        返回给 register_forward_hook 的回调。

        例: linear1 算完 y 之后:
          1) 丢掉完整 [128,64]，写回分片 [64,64]
          2) 按讲义预取「后面第 2 个被切层」的 all-gather
             index=1 → 预取 _sharded_modules[3] == lm_head
        """

        def hook(module: nn.Module, inputs: tuple, output: Tensor) -> None:
            meta: ShardMeta = self._meta[id(mod)]
            i: int = meta.index  # 例: linear1 → 1
            self._free_full(mod)

            j: int = i + 2  # 例: 1+2=3
            if j < len(self._sharded_modules):
                # 例: 对 lm_head 发 async all-gather；收据进该层 meta.all_gather_work
                # 等真正进入 lm_head 的 forward_pre 再 wait
                self._start_all_gather(self._sharded_modules[j], async_op=True)

        return hook

    def _make_backward_pre_hook(self, mod: nn.Module):
        """
        返回给 register_full_backward_pre_hook 的回调。

        例: autograd 马上要对 linear1 做反向时 → 再次装上完整权重，
            这样 einsum / embedding 反传读到的 Parameter.data 才是对的。

        混合精度特例（对齐 test_fsdp 基线）:
          Linear backward → 仍用 compute_dtype（例 fp16）完整权重
          Embedding backward → 强制 fp32 完整权重
        """

        def hook(module: nn.Module, grad_output: tuple) -> None:
            if isinstance(mod, Embedding) and self.compute_dtype is not None:
                self._wait_and_install_full(mod, force_dtype=torch.float32)
            else:
                self._wait_and_install_full(mod)

        return hook

    def _make_grad_hook(self, mod: nn.Module):
        """
        返回给 register_post_accumulate_grad_hook 的回调。

        触发时机: 该 Parameter 的 .grad 刚被累加完。

        例 linear1、world_size=2:
          进入时: param 仍是完整形态，param.grad.shape == [128, 64]
                 （数值 = 本卡 local batch 上的局部梯度）
          过程中: async reduce-scatter
                 rank0 最终得到「各卡在行 0:64 上求和」→ shape [64, 64]
                 rank1 得到行 64:128 的和 → shape [64, 64]
          离开时: param（weight）已缩回分片；param.grad 指向分片缓冲（可能还在传）
        """

        def hook(param: Tensor) -> None:
            # param is mod.weight（同一个对象）
            if param.grad is None:
                return

            meta: ShardMeta = self._meta[id(mod)]

            # 先拷出完整局部梯度。例: full_grad.shape == [128, 64], float32
            full_grad: Tensor = (
                param.grad.detach().to(torch.float32).contiguous().clone()
            )

            # PyTorch 要求: .grad 的 shape/dtype 必须和当前 param.data 一致。
            # 所以必须先把 weight 缩回分片，再挂上分片形状的 .grad。
            param.grad = None
            self._free_full(mod)
            # 此刻: param.shape == [64, 64]（例）

            # 例: shard_shape == (64, 64)
            shard_shape: tuple[int, ...] = (
                full_grad.shape[: meta.shard_dim]
                + (meta.shard_size,)
                + full_grad.shape[meta.shard_dim + 1 :]
            )
            # 例: 空的 [64, 64]，reduce-scatter 写完后才是各卡 SUM
            shard_grad: Tensor = full_grad.new_empty(shard_shape)

            work: dist.Work | bool | None = dist.reduce_scatter_tensor(
                shard_grad,
                full_grad,
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            assert isinstance(work, dist.Work)
            # 例: self._grad_works 末尾多一个「未完成的 reduce-scatter」收据
            self._grad_works.append(work)

            # 先挂上缓冲；finish_gradient_synchronization 里 wait 之后数值才最终正确
            # 例: param.grad.shape == [64, 64], dtype float32
            param.grad = shard_grad

        return hook

    def _replicated_grad_hook(self, param: Tensor) -> None:
        """
        未切开的参数（例 norm1.weight）用 all-reduce，不是 reduce-scatter。

        例: 每卡 param.grad.shape == [64]
            async all-reduce SUM → 各卡得到相同的总和
            finish 里再 div_(world_size) 得到平均
        """
        if param.grad is None:
            return
        work: dist.Work | bool | None = dist.all_reduce(
            param.grad,
            op=dist.ReduceOp.SUM,
            async_op=True,
        )
        assert isinstance(work, dist.Work)
        self._grad_works.append(work)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        例: logits = fsdp_model(input_ids)
          → 转发给 ToyFSDPModel.forward
          → 途中每个被切层的 forward_pre / forward_post 自动跑通信逻辑
        """
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """
        在 loss.backward() 之后、optimizer.step() 之前调用。

        做什么:
          1) 等待所有异步 reduce-scatter / all-reduce 完成
          2) 把 SUM 除以 world_size → 平均梯度

        不做什么:
          不更新权重。更新发生在之后的 optimizer.step()。

        例 world_size=2、linear1:
          wait 前: param.grad 可能仍是「正在写入的 SUM 缓冲」
          wait 后: param.grad 是两卡局部梯度之和，shape [64, 64]
          div_ 后: 平均梯度，可直接给 SGD/AdamW
        """
        for work in self._grad_works:
            work.wait()
        self._grad_works.clear()

        world_size: int = self.world_size
        seen: set[int] = set()
        for param in self.module.parameters():
            pid: int = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            if param.grad is not None:
                # 例被切: param.grad.shape == [64, 64]
                # 例 norm:  param.grad.shape == [64]
                param.grad.div_(world_size)

    def gather_full_params(self) -> dict[str, Tensor]:
        """
        测试用：把分片 all-gather 回完整参数，方便和单卡模型比对。

        例返回:
          {
            "embedding.weight": Tensor[100, 64],  # 由两卡分片拼回
            "norm1.weight":     Tensor[64],       # 未切，直接 clone
            "linear1.weight":   Tensor[128, 64],
            "norm2.weight":     Tensor[128],
            "linear2.weight":   Tensor[64, 128],
            "lm_head.weight":   Tensor[100, 64],
          }
        """
        # 先保证大家都处在「分片常驻」状态，避免 all-gather 读到半截完整视图
        for mod in self._sharded_modules:
            meta: ShardMeta = self._meta[id(mod)]
            if meta.all_gather_work is not None:
                meta.all_gather_work.wait()
                meta.all_gather_work = None
            if meta.is_full:
                self._free_full(mod)
            meta.full = None

        out: dict[str, Tensor] = {}
        # 例: id(linear1.weight) → linear1 模块
        id_to_mod: dict[int, nn.Module] = {
            id(mod.weight): mod for mod in self._sharded_modules
        }

        for name, param in self.module.named_parameters():
            if id(param) in id_to_mod:
                mod = id_to_mod[id(param)]
                meta = self._meta[id(mod)]
                # 例: shard.shape == [64, 64]
                shard: Tensor = param.data.contiguous()
                # 例: full 最终 == [128, 64]
                full: Tensor = shard.new_empty(meta.full_shape)
                dist.all_gather_into_tensor(full, shard, async_op=False)
                out[name] = full.detach().to(torch.float32).clone()
            else:
                # 例: name == "norm1.weight" → 不切，原样返回
                out[name] = param.data.detach().to(torch.float32).clone()
        return out
