# =============================================================================
# 讲义 Problem (distributed_communication_single_node)
# 单机多进程：benchmark all-reduce 耗时
# =============================================================================
#
# 作业要求（摘要）：
#   - 张量：float32
#   - 数据量：1MB / 10MB / 100MB / 1GB
#   - 进程数（= GPU 数）：2 / 4 / 6
#   - 正式结果用 NCCL + GPU；本机可用 Gloo + CPU 冒烟
#
# 讲义 §5.1.1 计时注意点（本脚本都按这个做）：
#   1) 先 warmup 几轮（默认 5），再正式计时
#   2) GPU 上即使 async_op=False，也只保证「通信入队」，
#      所以计时前后都要 torch.cuda.synchronize()
#   3) 不同 rank 测到的时间会略有差别 → all_gather_object 收齐再汇总
#
# 本文件只负责「测 + 写 CSV」，不做图、不写报告 Markdown。
# =============================================================================

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 作业里写的「1MB」按 10^6 字节理解。
_MB = 1000 * 1000

# 默认扫的数据量（单位：MB）。1 GB = 1000 MB。
_DEFAULT_SIZES_MB = (1, 10, 100, 1000)


@dataclass
class BenchRow:
    """一行结果：某一个 (world_size, 数据量) 配置的汇总统计。"""

    backend: str
    world_size: int
    size_mb: int
    numel: int
    bytes: int
    warmup: int
    iters: int
    # 各 rank 自己测到的「平均单次耗时」再跨 rank 汇总：
    latency_ms_mean: float  # 跨 rank 再平均（主指标）
    latency_ms_min: float  # 最快的那个 rank
    latency_ms_max: float  # 最慢的那个 rank（通信常被最慢的卡住）
    # 粗算「算法带宽」= 张量字节数 / 平均耗时。
    # 注意：真实 all-reduce 在环上要传大约 2*(N-1)/N 倍数据，
    # 所以这只是方便看随 size 怎么涨的参考值，不是硬件峰值带宽。
    approx_alg_bandwidth_GBps: float


def _size_bytes_to_numel(size_bytes: int) -> int:
    """float32 每个元素 4 字节 → 元素个数 = 字节数 // 4。"""
    if size_bytes % 4 != 0:
        raise ValueError(f"size_bytes={size_bytes} 不能被 4 整除（float32）")
    return size_bytes // 4


def setup(rank: int, world_size: int, backend: str, master_port: str) -> torch.device:
    """
    每个 worker 进程一进来先跑这里：绑定设备 + 建进程组。

    返回值：这个 rank 以后创建张量该用的 device。
    """
    # 所有进程必须连同一个 master（单机就用 localhost + 同一个端口）。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port

    if backend == "nccl":
        # 讲义推荐：rank i 固定用第 i 张 GPU，避免多个进程抢同一张卡。
        # set_device 之后，裸写 tensor.to("cuda") 也会落到这张卡上。
        if not torch.cuda.is_available():
            raise RuntimeError("backend=nccl 需要 CUDA，但当前 torch.cuda.is_available()=False")
        n_gpu = torch.cuda.device_count()
        if world_size > n_gpu:
            raise RuntimeError(
                f"rank={rank}: 需要 {world_size} 张 GPU，但只看到 {n_gpu} 张。"
                "（不要把多进程挤到同一张卡上假装多卡，结果没有参考价值。）"
            )
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    elif backend == "gloo":
        # Gloo：CPU 张量就能通信，方便没多卡时先把脚本跑通。
        device = torch.device("cpu")
    else:
        raise ValueError(f"不支持的 backend={backend!r}，请用 nccl 或 gloo")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup() -> None:
    """先 barrier 对齐，再销毁进程组，避免有的 rank 先退出导致别人挂住。"""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _synchronize(device: torch.device) -> None:
    """
    把「异步入队的 CUDA 工作」等到真正做完。

    CPU（gloo）路径：什么都不用做，通信本身就是在 CPU 上阻塞完成的。
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench_one_size(
    *,
    tensor: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> float:
    """
    对「已经造好的同一块张量」做 warmup + 正式计时。

    返回：这个 rank 上，正式 iters 次 all-reduce 的平均耗时（秒）。
    """
    # ----- warmup：不计时 -----
    # NCCL 第一次调用常会偏慢（建通信域、选算法等），所以先扔掉几轮。
    for _ in range(warmup):
        dist.all_reduce(tensor, async_op=False)
        _synchronize(device)

    # ----- 正式计时 -----
    # 用 perf_counter（墙上时钟）+ synchronize：
    #   起点：保证 GPU 上之前的活干完
    #   all_reduce：把通信 kernel 丢进 GPU 队列（async_op=False 也不等于通信已结束）
    #   终点：再 synchronize，等到这次通信真正完成
    # 这样测到的才是「这次 all-reduce 端到端多久」。
    _synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(tensor, async_op=False)
        _synchronize(device)
    t1 = time.perf_counter()

    return (t1 - t0) / iters


def worker(
    rank: int,
    world_size: int,
    backend: str,
    sizes_mb: list[int],
    warmup: int,
    iters: int,
    master_port: str,
    out_csv: str,
) -> None:
    """
    mp.spawn 启动的每个子进程入口。

    签名必须是 (rank, ...)：spawn 会自动把 rank 塞进第一个参数。
    """
    device = setup(rank, world_size, backend, master_port)

    # 这个进程本地攒的「每个 size 的平均耗时（秒）」。
    # 稍后用 all_gather_object 跟别的 rank 交换，才能算跨 rank 的 mean/min/max。
    local_mean_s: list[float] = []
    meta: list[tuple[int, int, int]] = []  # (size_mb, numel, nbytes)

    try:
        for size_mb in sizes_mb:
            nbytes = size_mb * _MB
            numel = _size_bytes_to_numel(nbytes)

            # 每个 rank 各自持有一份同样 shape 的 float32 数据。
            # all-reduce 默认做「求和」：测延迟时数值内容不重要，随机即可。
            tensor = torch.randn(numel, dtype=torch.float32, device=device)

            mean_s = bench_one_size(
                tensor=tensor,
                device=device,
                warmup=warmup,
                iters=iters,
            )
            local_mean_s.append(mean_s)
            meta.append((size_mb, numel, nbytes))

            # 及时释放，避免扫到 1 GB 时显存/内存叠太多份残留。
            del tensor
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # 把每个 rank 的 local_mean_s 收集到「每人都有一份完整列表」里。
        # all_gather_object：可以传 Python 对象（list/float），比 tensor all_gather 省事。
        gathered: list[list[float] | None] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_mean_s)

        # 只有 rank 0 负责打印和写文件，避免 6 个进程一起刷屏/抢着写 CSV。
        if rank == 0:
            assert all(g is not None for g in gathered)
            rows: list[BenchRow] = []
            for i, (size_mb, numel, nbytes) in enumerate(meta):
                # gathered[r][i] = rank r 在第 i 个 size 上测到的平均秒数
                per_rank_ms = [gathered[r][i] * 1e3 for r in range(world_size)]  # type: ignore[index]
                mean_ms = sum(per_rank_ms) / len(per_rank_ms)
                min_ms = min(per_rank_ms)
                max_ms = max(per_rank_ms)
                bw = (nbytes / (mean_ms / 1e3)) / 1e9  # 字节/秒 → GB/s
                rows.append(
                    BenchRow(
                        backend=backend,
                        world_size=world_size,
                        size_mb=size_mb,
                        numel=numel,
                        bytes=nbytes,
                        warmup=warmup,
                        iters=iters,
                        latency_ms_mean=mean_ms,
                        latency_ms_min=min_ms,
                        latency_ms_max=max_ms,
                        approx_alg_bandwidth_GBps=bw,
                    )
                )

            _print_table(rows)
            _append_csv(Path(out_csv), rows)
            print(f"[rank0] 已追加写入 CSV: {out_csv}")
    finally:
        cleanup()


def _print_table(rows: list[BenchRow]) -> None:
    """人眼友好的小表；正式存档以 CSV 为准。"""
    hdr = (
        f"{'backend':>6} {'N':>3} {'MB':>5} {'ms_mean':>10} "
        f"{'ms_min':>10} {'ms_max':>10} {'GB/s~':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.backend:>6} {r.world_size:>3} {r.size_mb:>5} "
            f"{r.latency_ms_mean:>10.3f} {r.latency_ms_min:>10.3f} "
            f"{r.latency_ms_max:>10.3f} {r.approx_alg_bandwidth_GBps:>8.3f}"
        )


def _append_csv(path: Path, rows: list[BenchRow]) -> None:
    """追加写 CSV；文件不存在时先写表头。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark single-node all-reduce (CS336 distributed_communication_single_node)."
    )
    p.add_argument(
        "--world-size",
        type=int,
        nargs="+",
        default=[2, 4],
        help="要测的进程数/GPU 数列表。默认：2 4。每次只 spawn 其中一个 N。",
    )
    p.add_argument(
        "--backend",
        choices=("nccl", "gloo"),
        default="nccl",
        help="正式多卡用 nccl；本机没多卡时可改 gloo 做流程冒烟。",
    )
    p.add_argument(
        "--sizes-mb",
        type=int,
        nargs="+",
        default=list(_DEFAULT_SIZES_MB),
        help="数据量列表，单位 MB（10^6 字节）。默认：1 10 100 1000。",
    )
    p.add_argument("--warmup", type=int, default=5, help="每个配置正式计时前的 warmup 次数。")
    p.add_argument("--iters", type=int, default=20, help="每个配置正式计时的重复次数（取平均）。")
    p.add_argument(
        "--master-port",
        default="29501",
        help="进程组 master 端口（避开 hello-world 常用的 29500）。",
    )
    p.add_argument(
        "--output",
        default="artifacts/all_reduce_single_node.csv",
        help="结果 CSV 路径（追加写入）。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 连续 spawn 多个 world_size：一轮完全结束后再开下一轮，避免端口/进程组打架。
    for world_size in args.world_size:
        if world_size < 1:
            raise ValueError(f"world_size 必须 ≥1，收到 {world_size}")

        if args.backend == "nccl":
            n_gpu = torch.cuda.device_count()
            if n_gpu < world_size:
                raise RuntimeError(
                    f"要测 world_size={world_size}，但当前只看到 {n_gpu} 张 GPU。"
                    f"请换多卡机器，或先只跑：--world-size {n_gpu}（且 n_gpu∈{{2,4,6}} 才对齐作业）。"
                )

        print(
            f"\n=== spawn world_size={world_size} backend={args.backend} "
            f"sizes_mb={args.sizes_mb} warmup={args.warmup} iters={args.iters} ==="
        )

        # mp.spawn：起 world_size 个进程，每个跑 worker(rank, world_size, ...)。
        mp.spawn(
            fn=worker,
            args=(
                world_size,
                args.backend,
                list(args.sizes_mb),
                args.warmup,
                args.iters,
                args.master_port,
                args.output,
            ),
            nprocs=world_size,
            join=True,
        )

    print(f"\n全部跑完。结果在: {args.output}")


if __name__ == "__main__":
    # Linux 下 torch 分布式常用 spawn 启动方式；显式设一下更稳。
    mp.set_start_method("spawn", force=True)
    main()
