# =============================================================================
# FlashAttention-2 Triton 前向：术语表（粘贴备用）
# =============================================================================
#
# ----- 数学量（Algorithm 1）-----
# Q, K, V     查询 / 键 / 值。HBM 上的输入张量，形状 (batch, N_q|N_k, d)。
# O           注意力输出。HBM 上，形状 (batch, N_q, d)。
# S / Sij     缩放分数 QK^T/sqrt(d)；Sij 是当前 query tile × key tile 的子块。
#             只在片上出现，不写回 HBM。
# P_tilde     未归一化 softmax 分子 exp(Sij - m_i)，记作 P̃。只在片上。
# m_i         行方向运行最大值（小写 m）。片上，形状 (B_q,)。初值 -inf。
# l_i         softmax 分母的运行累加（小写 l）。片上，形状 (B_q,)。初值 0。
#             尚无 log；扫完所有 key tile 后才和 m_i 一起得到大 L。
# L           logsumexp（大写 L）。HBM 上，形状 (batch, N_q)。
#             L = m_i + log(l_i)。给反向 / 测试用。不是小 l_i。
# d           特征维（最后一维）。缩放 scale = 1/sqrt(d)。与 l、L 无关。
# B_q, B_k    query / key 的 tile 大小（代码里常叫 Q_TILE_SIZE / K_TILE_SIZE）。
# N_q, N_k    query / key 序列长度（代码里常叫 N_QUERIES / N_KEYS）。
# T_q, T_k    tile 个数：ceil(N_q/B_q)、ceil(N_k/B_k)。
# scale       1/sqrt(d)，传入 kernel 的标量。
# is_causal   因果 mask 开关（B 可先忽略；C 再做）。
#
# ----- 大小写易混（钉死）-----
# 小 l / l_i  片上分母累加器。不分配 HBM，没有 L_ptr，没有 stride_l*。
# 大 L        HBM 上的 logsumexp 张量。有 L_ptr、stride_lb、stride_lq。
# 名字里的 l  Triton 参数 L_ptr / stride_lb / stride_lq 的 “l” 指大写 L 张量，
#             只是标识符用了小写 l 当前缀，不要当成小 l_i。
#
# ----- GPU / Triton 执行模型 -----
# HBM         GPU 片外大显存。Q/K/V/O/L 整张住在这里；load/store 相对慢。
# 片上        寄存器 / SRAM。快但小；m_i、l_i、O_i、Sij、P_tilde 住这里。
# launch grid 启动时的并行形状，本作业为 (T_q, batch)。
# program     grid 上的一份内核实例（约等于 CUDA 一个 thread block 级工作单元）。
# program_id(0)  本 program 的 query tile 下标，记 query_tile_index，范围 0..T_q-1。
# program_id(1)  本 program 的 batch 下标，记 batch_index，范围 0..batch-1。
# 一个 program 的职责：固定 (batch_index, query_tile_index)；只读该 Q tile；
#             内层串行扫全部 key tile；只写对应的 O tile 与 L tile。
#
# ----- 指针与步长（把逻辑下标翻译成 HBM 元素偏移）-----
# *_ptr       张量在 HBM 的基址。Q_ptr/K_ptr/V_ptr/O_ptr/L_ptr。
# stride      某轴下标 +1 时，内存里跳过多少个【元素】（不是 byte）。
#
# Q 的三轴步长（Q 形状 batch × N_q × d）：
#   stride_qb   batch 轴 +1 的步长。典型连续布局 ≈ N_q * d。
#   stride_qq   query 轴 +1 的步长。典型 ≈ d。
#   stride_qd   特征维 d +1 的步长。典型 ≈ 1。
#   元素偏移： b*stride_qb + i*stride_qq + c*stride_qd。
#
# K 的三轴步长（batch × N_k × d）：
#   stride_kb, stride_kk, stride_kd   同理，轴是 batch / key / d。
#
# V 的三轴步长：
#   stride_vb, stride_vk, stride_vd
#
# O 的三轴步长（与 Q 同形状）：
#   stride_ob, stride_oq, stride_od
#
# L 的两轴步长（L 形状 batch × N_q；这里的 l 前缀 = 大写 L 张量）：
#   stride_lb   L 的 batch 轴 +1。典型连续布局 ≈ N_q。
#   stride_lq   L 的 query 轴 +1。典型 ≈ 1。
#   元素偏移： b*stride_lb + i*stride_lq。
#   再次强调：没有 “小 l_i 的 stride”，小 l_i 不进 HBM。
#
# Host 侧取值：
#   Q.stride(0/1/2) → stride_qb/qq/qd，K/V/O 同理；L.stride(0/1) → stride_lb/lq。
#
# ----- 块指针（make_block_ptr）：登记 “从 HBM 哪一块 load/store” -----
# base        如 Q_ptr + batch_index * stride_qb：先跳到本 batch，再当二维矩阵看。
# shape       当前 batch 视图下的逻辑全形状，如 Q/O 为 (N_q, d)，K/V 为 (N_k, d)，
#             L 为 (N_q,)。
# strides     该二维（或一维）视图内各轴步长；batch 步长已进 base，这里不再出现。
# offsets     本块起点。Q/O/L：行起点 = query_tile_index * Q_TILE_SIZE；
#             K/V：从 (0,0) 起，靠 advance 滑到后续 key tile。
# block_shape 本块大小。Q/O：(Q_TILE_SIZE, D)；K/V：(K_TILE_SIZE, D)；
#             L：(Q_TILE_SIZE,)。
# order       告诉编译器哪维更连续，便于生成访存；按讲义用 (1,0) 或 L 的 (0,)。
# advance     如 K_block_ptr.advance((K_TILE_SIZE, 0))：行起点 +B_k，窗口移到下一块 key。
#
# ----- 片上 ↔ HBM 数据运动 -----
# tl.load(block_ptr)   HBM → 片上。
# tl.store(block_ptr, x) 片上 → HBM。
# tl.dot(A, B)         片上矩阵乘；Sij 用 tl.dot(Q_i, tl.trans(K_j))*scale；
#                      O 更新用 tl.dot(P_tilde, V_j, acc=O_i)。
# tl.trans             转置；让 K_j 从 (B_k,d) 变成 (d,B_k) 以便和 Q_i 相乘。
# constexpr            编译期常量（D、Q_TILE_SIZE、K_TILE_SIZE），便于生成专用代码。
#
# ----- 精度（讲义要求）-----
# 片上 O_i / l_i / m_i 用 tl.float32。
# P_tilde @ V 前：P_tilde 转到 V 的 dtype。
# 写回 O 前：转到 O_block_ptr.type.element_ty（与输入 dtype 一致）。
# L 通常以 float32 写在 HBM。
#
# ----- autograd 包装 -----
# ctx                      forward 的上下文对象（储物柜）。
# ctx.save_for_backward    把 L,Q,K,V,O 挂上，供反向 / 测试从 saved_tensors 取 L。
# forward 返回值           只返回 O；大 L 不直接 return，而在 saved_tensors 里。
#
# D            特征维（即数学里的小写 d）。kernel 里作 tl.constexpr，习惯大写。
#              Q/K/V/O 最后一维长度；与小 l、大 L 无关。
# program_id   launch grid 上的坐标。本作业 grid=(T_q, batch)：
#              program_id(0)=query_tile_index，program_id(1)=batch_index（即 01 顺序）。
#              需要两维是因为 batch 与 query tile 都要并行；顺序 01 是讲义约定，不是硬件强制。
#              小 l_i 无 program_id；它在单个 program 内部随 key tile 串行更新。
# =============================================================================

import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 沿哪些维度给 index，就意味着哪些维度是可以进行的
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)

    Q_i = tl.load(Q_block_ptr)

    # 沿 key tile 串行滑窗；key_tile_index 给 causal 下标用
    # 有一个很重要的认知是：沿哪个维度滑窗，就意味着这个维度是要串行的
    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        Sij = tl.dot(Q_i.to(tl.float32), tl.trans(K_j.to(tl.float32))) * scale

        # -----------------------------------------------------------------
        # 因果 mask（is_causal）——为什么要有、做什么、语法怎么读
        # -----------------------------------------------------------------
        #
        # 【为什么要有这个 flag】
        # 语言模型生成时，位置 t 的 token 只能看见位置 0..t 的过去，
        # 不能看见 t+1.. 的未来。注意力里这叫因果（causal）约束：
        #   对第 q 个 query、第 k 个 key：若 k > q（key 在未来），该分数必须作废。
        # 训练/推理有时要双向注意力（BERT 一类），有时要因果注意力（GPT 一类），
        # 所以用一个布尔开关 is_causal：
        # 这个 flag 从 Python 的 forward(..., is_causal=...) 传进 kernel；
        # 类型写成 is_causal: tl.constexpr，表示编译期常量：True/False 两套代码
        # 路径可以分别特化，运行时不必每次再分支得很贵。
        # 同时还要 ctx.is_causal = is_causal，把开关存进 autograd 上下文，
        # 反向时才能知道前向有没有 mask（否则梯度会对错位置）。
        #
        # 【做了之后有什么效果】
        # Softmax 前：把“未来”位置的分数加上 -1e6（变得极小）。
        # Softmax 后：这些位置的权重 ≈ 0，等于没看见未来。
        # 保留位置（k <= q）分数不变，行为与普通注意力相同。
        #
        # 【本段在算什么】
        # 当前 program 只拿着一块 query（B_q 行）和一块 key（B_k 列），
        # Sij 形状是 (B_q, B_k)。我们要一张同形状的 True/False 表：
        #   True  = 允许（key 不在未来）
        #   False = 屏蔽（key 在未来）→ 给 Sij 对应元素 +(-1e6)
        #
        if is_causal:
            # q_idx：本块里每一行 query 的【全局序列下标】。
            #   query_tile_index * Q_TILE_SIZE = 本块第一行在整段序列里的起点
            #   tl.arange(0, Q_TILE_SIZE)     = [0, 1, ..., B_q-1]，块内局部行号
            #   相加后例如 tile=1、B_q=16 → q_idx = [16, 17, ..., 31]
            # 一维向量，长度 B_q。
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

            # k_idx：本块里每一列 key 的【全局序列下标】，同理。
            # 一维向量，长度 B_k。
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)

            # 把两个一维向量“拉开”成 B_q × B_k 的比较表（广播）。
            #
            #   q_idx[:, None]
            #     q_idx 本是形状 (B_q,)
            #     [:, None] 读作：保留第 0 维全部（冒号 :），再在末尾插入长度为 1 的新维（None）
            #     结果形状 (B_q, 1) —— “一列竖着的 query 下标”
            #   k_idx[None, :]
            #     [None, :] = 在开头插一维，再保留原来那一维全部
            #     结果形状 (1, B_k) —— “一行横着的 key 下标”
            #   两者做 >= 时，(B_q,1) 与 (1,B_k) 自动广播成 (B_q, B_k)：
            #     格子 (r,c) 存的是：q_idx[r] >= k_idx[c] ?
            #   这就是讲义说的 “square mask of size B_q × B_k”
            #
            # 因果规则：允许 key 下标 <= query 下标（含自己），即 q >= k。
            causal_mask = q_idx[:, None] >= k_idx[None, :]

            # tl.where(条件, 真时取值, 假时取值)：逐元素三元选择。
            #   条件为 True  → 保持原 Sij（可见的过去/现在）
            #   条件为 False → Sij + (-1e6)（未来位置，softmax 后≈0）
            # 写成 Sij + (-1.0e6) 而不是单独的常量，是为了“在原分数上加惩罚”，
            Sij = tl.where(causal_mask, Sij, Sij + (-1.0e6))

        m_i_prev = m_i
        m_i = tl.maximum(m_i, tl.max(Sij, axis=-1))

        P_tilde = tl.exp(Sij - m_i[:, None])
        alpha = tl.exp(m_i_prev - m_i)

        l_i = alpha * l_i + tl.sum(P_tilde, axis=-1)
        O_i = O_i * alpha[:, None]
        O_i = tl.dot(P_tilde.to(V_j.dtype), V_j, acc=O_i)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i.to(L_block_ptr.type.element_ty))


def _choose_flash_tiles(seq_len: int, d: int) -> tuple[int, int]:
    # A800 shared-memory limited: large (B_q,d) tiles OOM the kernel.
    # Keep tiles modest; grow only when d is small.
    if d >= 128:
        return 16, 16
    if d >= 64:
        return (32, 32) if seq_len >= 2048 else (16, 16)
    if seq_len >= 8192:
        return 64, 64
    if seq_len >= 1024:
        return 32, 32
    return 16, 16


# =============================================================================
# Algorithm 2：Triton 反向
# =============================================================================
#
# ## 1. 数学：三个梯度都是「分块求和」
#
#   dQ[i] = Σ_j  dS[i,j] @ K[j] / √d     （固定 query 块 i，沿 key 块 j 累加）
#   dK[j] = Σ_i  dS[i,j]^T @ Q[i] / √d   （固定 key 块 j，沿 query 块 i 累加）
#   dV[j] = Σ_i  P[i,j]^T @ dO[i]          （固定 key 块 j，沿 query 块 i 累加）
#
# 其中一块 (i,j) 上：
#   S=Q_i K_j^T/√d,  P=exp(S-L_i),  dP=dO_i V_j^T,  dS=P∘(dP-D_i)
#
# ## 2. 核心矛盾：一次并行划分，做不到三个梯度都「单写者」
#
# GPU 原则：同一块 HBM 地址最好只让一个 program 写。
# 多 program 同时「读旧值 → 加贡献 → 写回」同一地址时，后写覆盖先写，少算一项。
#
# 例子：8 query→4 块 Q0..Q3；8 key→4 块 K0..K3。
#
# 方案1——只开一个 kernel、按 Q 并行（4 个 program）：
#   ✅ dQ[0]：只有 program_Q0 写
#   ❌ dK[0] = Σ_{i=0..3} (...) ：program_Q0..Q3 都要往 dK[0] 加 → 4 写者冲突
#   ❌ dV[0] 同理
#
# 方案2——只开一个 kernel、按 K 并行（4 个 program）：
#   ❌ dQ[0] = Σ_j (...) ：4 个 key-program 都要往 dQ[0] 加 → 冲突
#   ✅ dK[0]、dV[0]：只有 program_K0 写
#
# 结论：按 Q 分则 dQ 安全、dK/dV 危险；按 K 分则相反。必须拆成两个 kernel。
#
# ## 3. 设计：两阶段，各自单写者
#
# 阶段 A flash_bwd_dkdv_kernel：按 K 并行，开 T_k 个 program（再 ×batch）。
#   program_Kj 独占 K_j，内层扫全部 Qi，片上累加后只 store dK[j]、dV[j]。
#
# 阶段 B flash_bwd_dq_kernel：按 Q 并行，开 T_q 个 program（再 ×batch）。
#   program_Qi 独占 Q_i，内层扫全部 Kj，片上累加后只 store dQ[i]。
#
# ## 4. 为什么 P 算两遍
#
# 每个 (Qi,Kj) 在阶段 A 算一次 P，阶段 B 再算一次。
# 原因：阶段 A 的 P 活在 program_Kj 的片上；阶段 B 的 program_Qi 读不到。
# 若把全部 P tile 写回 HBM 再读，等于又存了 N×N 注意力矩阵，违背 Flash 初衷。
# 小块 QK^T+exp 在 SRAM 里很快；重算一次，换「不存大 P、无跨 block atomic」。
#
# ## 5. program_id(0) / program_id(1) 从哪来（不是编译器随便编的）
#
# 宿主 Python 里 launch，例如：
#   flash_bwd_dkdv_kernel[(T_k, batch)](...)
# 方括号里的 (T_k, batch) 叫 launch grid：GPU 启动 T_k×batch 份同一内核。
# Triton 运行时给每一份发一个二维坐标：
#   tl.program_id(0) ∈ {0,...,T_k-1}   ← grid 第 0 维（我们约定 = key tile）
#   tl.program_id(1) ∈ {0,...,batch-1} ← grid 第 1 维（我们约定 = batch）
# 顺序 01 完全由「我们写的 grid 元组顺序」决定；换成 [(batch, T_k)] 就要对调两个
# program_id 的用法。内核里读 program_id，只是在问：「我是这份并行里的哪一格？」
#
# =============================================================================


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 阶段 A：算 dK、dV。
    # program_id 来自宿主 launch：flash_bwd_dkdv_kernel[(T_k, batch)](...)
    #   grid 第 0 维长度 = T_k  → program_id(0) = 第几个 key tile（0..T_k-1）
    #   grid 第 1 维长度 = batch → program_id(1) = 第几个 batch
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # 内层从 query tile 0 起：Q / dO / L / D_vec 每轮 advance 一块
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    # D_ptr 这里是向量 D=rowsum(O∘dO)，不是特征维；命名 D_vec 避免和 constexpr D 混
    Dvec_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES,),
        strides=(stride_Dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # dK_0 = 0, dV_0 = 0；后面每一轮 i：acc = acc + tile_i 的贡献
    dK_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_acc = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

    K_j = tl.load(K_block_ptr).to(tl.float32)
    V_j = tl.load(V_block_ptr).to(tl.float32)

    for query_tile_index in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Q_i = tl.load(Q_block_ptr).to(tl.float32)
        dO_i = tl.load(dO_block_ptr).to(tl.float32)
        L_i = tl.load(L_block_ptr)                 # 大写 L：logsumexp
        D_i = tl.load(Dvec_block_ptr)              # D_i = sum_c O_ic * dO_ic

        # S_ij = Q_i K_j^T / sqrt(d)
        Sij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, Sij + (-1.0e6))

        # 第一次算 P：专供本阶段累加 dK/dV
        Pij = tl.exp(Sij - L_i[:, None])

        # dV += P^T @ dO
        dV_acc = tl.dot(tl.trans(Pij), dO_i, acc=dV_acc)

        # dP = dO @ V^T ； dS = P ∘ (dP - D)
        dPij = tl.dot(dO_i, tl.trans(V_j))
        dSij = Pij * (dPij - D_i[:, None])

        # dK += dS^T @ Q / sqrt(d)
        # 本轮贡献先乘 scale，再加进累加器（保持「total = prev + current」）
        dK_acc = dK_acc + tl.dot(tl.trans(dSij), Q_i) * scale

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        Dvec_block_ptr = Dvec_block_ptr.advance((Q_TILE_SIZE,))

    # 内层扫完所有 i：写出本 key tile 的最终 dK_j、dV_j
    tl.store(dK_block_ptr, dK_acc.to(dK_block_ptr.type.element_ty))
    tl.store(dV_block_ptr, dV_acc.to(dV_block_ptr.type.element_ty))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_Db, stride_Dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 阶段 B：算 dQ。
    # program_id 来自宿主 launch：flash_bwd_dq_kernel[(T_q, batch)](...)
    #   grid 第 0 维 = T_q    → program_id(0) = 第几个 query tile
    #   grid 第 1 维 = batch  → program_id(1) = 第几个 batch
    # 与阶段 A 相同机制；只是 grid 第 0 维从「按 K 分」换成「按 Q 分」。
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Dvec_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_Db,
        shape=(N_QUERIES,),
        strides=(stride_Dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # 内层从 key tile 0 起滑
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    # dQ_i^{(0)} = 0；每轮 j：dQ = dQ + dS_ij @ K_j / sqrt(d)
    dQ_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    Q_i = tl.load(Q_block_ptr).to(tl.float32)
    dO_i = tl.load(dO_block_ptr).to(tl.float32)
    L_i = tl.load(L_block_ptr)
    D_i = tl.load(Dvec_block_ptr)

    for key_tile_index in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        K_j = tl.load(K_block_ptr).to(tl.float32)
        V_j = tl.load(V_block_ptr).to(tl.float32)

        # 与阶段 A 同一套 S→P→dP→dS；这里是第二次算同一块 P
        Sij = tl.dot(Q_i, tl.trans(K_j)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            causal_mask = q_idx[:, None] >= k_idx[None, :]
            Sij = tl.where(causal_mask, Sij, Sij + (-1.0e6))

        Pij = tl.exp(Sij - L_i[:, None])
        dPij = tl.dot(dO_i, tl.trans(V_j))
        dSij = Pij * (dPij - D_i[:, None])

        # dQ += dS @ K / sqrt(d)
        dQ_acc = dQ_acc + tl.dot(dSij, K_j) * scale

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    tl.store(dQ_block_ptr, dQ_acc.to(dQ_block_ptr.type.element_ty))


class FlashAttention2TritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=0, B_k=0):
        # B_q/B_k <=0：走 _choose_flash_tiles；benchmark 可显式传入以做 tile 消融。
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        batch, N_q, d = Q.shape
        N_k = K.shape[1]
        if B_q is None or B_q <= 0 or B_k is None or B_k <= 0:
            B_q, B_k = _choose_flash_tiles(N_q, d)
        else:
            B_q, B_k = int(B_q), int(B_k)
        scale = 1.0 / math.sqrt(d)

        O = torch.empty_like(Q)
        L = torch.empty((batch, N_q), device=Q.device, dtype=torch.float32)

        T_q = triton.cdiv(N_q, B_q)

        flash_fwd_kernel[(T_q, batch)](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=bool(is_causal),
        )

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = bool(is_causal)
        ctx.B_q = B_q
        ctx.B_k = B_k
        return O

    @staticmethod
    def backward(ctx, grad_O):
        # Algorithm 2：宿主侧算 D，再 launch 两阶段 Triton kernel。
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        B_q, B_k = ctx.B_q, ctx.B_k
        dO = grad_O

        batch, N_q, d = Q.shape
        N_k = K.shape[1]
        scale = 1.0 / math.sqrt(d)

        # D = rowsum(O ∘ dO)
        D_vec = torch.sum(O.to(torch.float32) * dO.to(torch.float32), dim=-1)

        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        T_q = triton.cdiv(N_q, B_q)
        T_k = triton.cdiv(N_k, B_k)

        # 阶段 A：[(T_k, batch)] → program_id(0)=key tile, program_id(1)=batch
        flash_bwd_dkdv_kernel[(T_k, batch)](
            Q, K, V, dO, L, D_vec, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            dK.stride(0), dK.stride(1), dK.stride(2),
            dV.stride(0), dV.stride(1), dV.stride(2),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )

        # 阶段 B：[(T_q, batch)] → program_id(0)=query tile, program_id(1)=batch
        flash_bwd_dq_kernel[(T_q, batch)](
            Q, K, V, dO, L, D_vec, dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2),
            L.stride(0), L.stride(1),
            D_vec.stride(0), D_vec.stride(1),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            N_q, N_k,
            scale,
            D=d,
            Q_TILE_SIZE=B_q,
            K_TILE_SIZE=B_k,
            is_causal=is_causal,
        )
        # 对应 forward(Q,K,V,is_causal,B_q,B_k)
        return dQ, dK, dV, None, None, None
