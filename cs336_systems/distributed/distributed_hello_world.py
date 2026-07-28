# =============================================================================
# 讲义 §5.1：单机多进程 all-reduce 入门 demo（distributed_hello_world）
# =============================================================================
# 目标（讲义）：起 4 个 worker 进程，各自生成一个随机 int 张量，再 all-reduce 求和。
# 跑完后每个 rank 上的 data 都应变成「四个 rank 原始向量之和」（bitwise 相同）。
#
# 术语（后面作业会反复用）：
#   world_size  进程组里一共几个 worker（本 demo = 4）
#   rank        当前 worker 的整数 ID，取值 0 .. world_size-1；rank 0 常叫 master
#   process group  一组要互相通信的进程（init_process_group 建出来）
#   all-reduce  集合通信：各进程拿出自己的张量，按约定运算（默认求和）后，
#               每个进程都得到同一份完整结果（本例是 in-place 写回 data）
# =============================================================================

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int) -> None:
    # MASTER_ADDR / MASTER_PORT：进程组里的「联络点」。
    # 单机 demo 用本机回环地址；所有 worker 必须连同一个 master。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # 初始化进程组。backend="gloo"：CPU 上就能跑，方便本地调试。
    # 真正多卡训练一般用 "nccl"（只支持 CUDA 张量，通常更快）。
    # rank / world_size：告诉 PyTorch「我是谁、一共几个人」。
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank: int, world_size: int) -> None:
    # mp.spawn 会把 rank 作为第一个参数传进来：fn(rank, *args)。
    # 所以本函数签名必须是 (rank, world_size)。
    setup(rank, world_size)

    # 每个进程各自采样一个长度 3、元素在 [0,10) 的整型向量。
    # 不同 rank 种子/调度不同，before 打印出来一般不一样。
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")

    # all_reduce：把各 rank 的 data 按元素求和，结果写回每个 rank 自己的 data。
    # async_op=False：这次调用在通信完成（对 gloo/CPU 而言）后再返回。
    # （GPU + NCCL 时，False 也只保证「入队」，计时仍常要 cuda.synchronize。）
    dist.all_reduce(data, async_op=False)

    # after：四个 rank 应打印同一组数 = 各 rank before 向量的逐元素和。
    # 打印先后顺序不保证；只保证数值一致。
    print(f"rank {rank} data (after all-reduce): {data}")

    # 用完进程组要销毁，避免残留占用（小 demo 也养成习惯）。
    dist.destroy_process_group()


if __name__ == "__main__":
    # 一共起几个 worker；讲义示例是 4。
    world_size = 4

    # mp.spawn：一次启动 nprocs 个进程，每个跑 distributed_demo。
    # args=(world_size,) 会传给 fn，最终调用为：
    #   distributed_demo(rank, world_size)
    # join=True：等全部子进程结束，主进程才往下走。
    mp.spawn(
        fn=distributed_demo,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
