# 符号约定（对齐讲义 Algorithm 1 / 约第 25 页）：
#   Q, K, V : 查询 / 键 / 值，形状 (batch, N_q|N_k, d)
#   O       : 注意力输出，形状 (batch, N_q, d)
#   S       : 缩放后分数 Q K^T / sqrt(d)；子块记 Sij
#   P̃      : 未归一化 softmax 分子 exp(S - m)；代码里叫 P_tilde
#   m_i     : 行方向运行最大值（小写 m），形状 (batch, B_q)
#   l_i     : softmax 分母的运行累加（小写 l），尚无 log
#   L       : 最终 logsumexp（大写 L），L = m_i + log(l_i)，形状 (batch, N_q)
#   B_q,B_k : query / key tile 大小（≥16）
#   N_q,N_k : query / key 序列长度
#   d       : 隐藏维（最后一维），缩放用 1/sqrt(d)
#   T_q,T_k : tile 个数 ceil(N_q/B_q), ceil(N_k/B_k)
# 注意：小写 l_i ≠ 大写 L；d 是维数，和 l / L 无关。

import math
import torch
from einops import einsum


def attention_reference(Q, K, V, is_causal=False):
    # 朴素注意力，用来给 flash 实现做对照（benchmark baseline）。
    # S = Q K^T / sqrt(d)，可选 causal mask，P = softmax(S)，O = P V，L = logsumexp(S)。
    d = Q.shape[-1]
    N_q = Q.shape[-2]
    N_k = K.shape[-2]
    scale = 1.0 / math.sqrt(d)
    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        q_idx = torch.arange(N_q, device=Q.device)[None, :, None]
        k_idx = torch.arange(N_k, device=Q.device)[None, None, :]
        S = torch.where(q_idx >= k_idx, S, S.new_full((), -1e6))
    P = torch.softmax(S, dim=-1)
    O = einsum(P, V, "... q k, ... k d -> ... q d")
    L = torch.logsumexp(S, dim=-1)
    return O, L


def flash_attention_forward_pytorch(Q, K, V, B_q=16, B_k=16, is_causal=False):
    # 分块前向（Algorithm 1）：把 Q 沿 seq 维切成 T_q 块，K/V 切成 T_k 块。
    # 外层遍历 query 块，内层遍历 key 块，用 m_i / l_i / O_i 三个运行量把每块结果
    # 合并成和朴素前向一致的 O 和 L。S 和 P 始终留在片上，不写回 HBM。
    #
    # 数值上：无论输入是 fp32 还是 bf16，分块累加一律在 fp32 里做，最后再写回 Q.dtype。
    # 否则 bf16 的 V 与 fp32 的 P̃ 做 einsum 会直接报类型错误。

    batch = Q.shape[0]
    N_q = Q.shape[1]
    N_k = K.shape[1]
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # 最终要写回 HBM 的输出 O 和 logsumexp L。
    O = torch.empty_like(Q)
    L = torch.empty(batch, N_q, device=Q.device, dtype=torch.float32)

    # 外层：每次取一个 query 块 Q_i，形状 (batch, B_q, d)。
    for i in range(0, N_q, B_q):
        Q_i = Q[:, i : i + B_q, :].float()
        b_q = Q_i.shape[1]

        # 三个运行量，都按行（每个 query）维护。
        # m_i: 截至当前 key 块的行最大分数，初值 -inf。
        # l_i: 截至当前 key 块的 softmax 分母累加，初值 0。
        # O_i: 截至当前 key 块的加权和累加，最后再除以 l_i。
        O_i = torch.zeros(batch, b_q, d, device=Q.device, dtype=torch.float32)
        l_i = torch.zeros(batch, b_q, device=Q.device, dtype=torch.float32)
        m_i = torch.full((batch, b_q), float("-inf"), device=Q.device, dtype=torch.float32)

        # 内层：每次取一个 key/value 块，形状 (batch, B_k, d)。
        for j in range(0, N_k, B_k):
            K_j = K[:, j : j + B_k, :].float()
            V_j = V[:, j : j + B_k, :].float()

            # 当前子块分数 Sij，形状 (batch, B_q, B_k)。
            Sij = einsum(Q_i, K_j, "b q d, b k d -> b q k") * scale
            if is_causal:
                # 与 Triton / 朴素一致：全局下标 q < k 的位置置大负数，softmax ≈ 0。
                q_pos = torch.arange(i, i + b_q, device=Q.device)
                k_pos = torch.arange(j, j + K_j.shape[1], device=Q.device)
                Sij = torch.where(
                    q_pos[None, :, None] >= k_pos[None, None, :],
                    Sij,
                    Sij.new_full((), -1e6),
                )

            # 更新行最大值。Sij 最后一维是当前 key 块内的 B_k 个分数，
            # amax(dim=-1) 沿列方向约掉，得到这块每一行的最大值 (batch, B_q)，
            # 再和累积的 m_i 逐元素取 max。m_i_prev 留着重标定 l_i 和 O_i。
            m_i_prev = m_i
            m_i = torch.maximum(m_i, Sij.amax(dim=-1))

            # 当前块的未归一化 softmax 分子 P̃，形状 (batch, B_q, B_k)。
            # 减去新的 m_i 做数值稳定，和朴素 softmax 减去行最大值等价。
            P_tilde = torch.exp(Sij - m_i.unsqueeze(-1))

            # 重标定系数：旧最大值相对新最大值的 exp。形状 (batch, B_q)。
            # 之前累加的 l_i / O_i 是按旧 m 算的，换 m 之后要乘这个系数补回来。
            rescale = torch.exp(m_i_prev - m_i)

            # 分母累加：重标定后的旧 l_i + 当前块每行 exp 之和。
            l_i = rescale * l_i + P_tilde.sum(dim=-1)

            # 加权和累加：重标定后的旧 O_i + 当前块 P_tilde @ V_j。
            O_i = rescale.unsqueeze(-1) * O_i + einsum(P_tilde, V_j, "b q k, b k d -> b q d")

        # 内层扫完所有 key 块后：
        #   O_i / l_i → 写出 O
        #   m_i + log(l_i) → 写出大写 L（logsumexp）
        O[:, i : i + b_q, :] = (O_i / l_i.unsqueeze(-1)).to(Q.dtype)
        L[:, i : i + b_q] = m_i + torch.log(l_i)

    return O, L


def _flash_attention_backward_impl(Q, K, V, O, dO, L, is_causal: bool = False):
    # Eq.13–19：用存下的 L（及 O）重算 P，再反传；不依赖前向长期保存的大 P。
    # 全程在 fp32 里算，最后把梯度转回输入 dtype（支持 bf16）。
    out_dtype = Q.dtype
    Q = Q.float()
    K = K.float()
    V = V.float()
    O = O.float()
    dO = dO.float()
    L = L.float()

    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    N_q = Q.shape[-2]
    N_k = K.shape[-2]

    # (13) S = Q K^T / sqrt(d)
    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        q_idx = torch.arange(N_q, device=Q.device)[None, :, None]
        k_idx = torch.arange(N_k, device=Q.device)[None, None, :]
        S = torch.where(q_idx >= k_idx, S, S.new_full((), -1e6))

    # (14) P_ij = exp(S_ij - L_i)
    P = torch.exp(S - L.unsqueeze(-1))

    # D_i = rowsum(O ∘ dO)
    D = torch.sum(O * dO, dim=-1)

    # (15) dV = P^T dO
    dV = einsum(P, dO, "... q k, ... q d -> ... k d")
    # (16) dP = dO V^T
    dP = einsum(dO, V, "... q d, ... k d -> ... q k")
    # (17) dS_ij = P_ij (dP_ij - D_i)
    dS = P * (dP - D.unsqueeze(-1))
    # (18)(19) dQ = dS K / sqrt(d)，dK = dS^T Q / sqrt(d)
    dQ = einsum(dS, K, "... q k, ... k d -> ... q d") * scale
    dK = einsum(dS, Q, "... q k, ... q d -> ... k d") * scale
    return dQ.to(out_dtype), dK.to(out_dtype), dV.to(out_dtype)

# 作业要求：反向用普通 PyTorch + torch.compile（非 Triton）。
flash_attention_backward_pytorch = torch.compile(_flash_attention_backward_impl)


class FlashAttention2PyTorchFunc(torch.autograd.Function):
    # 把分块前向包进 autograd。测试调用方式：.apply(Q, K, V, is_causal)
    # 可选第 5/6 参：B_q, B_k（<=0 表示默认 16，benchmark 可显式传入以对齐 Triton tile）。

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, B_q=0, B_k=0):
        b_q = 16 if B_q is None or B_q <= 0 else int(B_q)
        b_k = 16 if B_k is None or B_k <= 0 else int(B_k)
        O, L = flash_attention_forward_pytorch(
            Q, K, V, B_q=b_q, B_k=b_k, is_causal=bool(is_causal)
        )

        # 把反向要用的张量挂到 ctx 上；测试也会从这里捞出形状 (batch, N_q) 的 L。
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = bool(is_causal)

        # forward 只返回 O；L 不直接返回。
        return O

    @staticmethod
    def backward(ctx, grad_O):
        L, Q, K, V, O = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward_pytorch(
            Q, K, V, O, grad_O, L, ctx.is_causal
        )
        # 对应 forward 的 (Q, K, V, is_causal, B_q, B_k)；后三者无梯度。
        return dQ, dK, dV, None, None, None


if __name__ == "__main__":
    torch.manual_seed(0)
    Q = torch.randn(2, 32, 64)
    K = torch.randn(2, 32, 64)
    V = torch.randn(2, 32, 64)

    O, L = attention_reference(Q, K, V)
    print("O.shape:", O.shape)
    print("L.shape:", L.shape)

    scale = 1.0 / math.sqrt(64)
    S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
    print("L 与 logsumexp(S) 一致:", torch.allclose(L, torch.logsumexp(S, dim=-1)))

    O_flash, L_flash = flash_attention_forward_pytorch(Q, K, V)
    print("flash O 与朴素一致:", torch.allclose(O_flash, O, atol=1e-4))
    print("flash L 与朴素一致:", torch.allclose(L_flash, L, atol=1e-4))