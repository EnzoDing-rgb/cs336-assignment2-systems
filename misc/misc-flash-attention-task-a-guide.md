# FlashAttention-2 任务 A 学习指南：纯 PyTorch 分块前向

**你要完成的事：** 实现 **Algorithm 1**（分块 + 在线 softmax），得到 $\mathbf{O}$ 和 $\mathbf{L}$，包进 `torch.autograd.Function`，在 `tests/adapters.py` 里注册，跑通 `uv run pytest -k test_flash_forward_pass_pytorch`。

**公式写法：** 行内 `$...$`，独立公式 `$$...$$`（前后各空一行）。

**本文只讲前向。** 任务 A 的 `backward` 填 `NotImplementedError` 即可。

---

## 一、术语表

下标 $i$：当前 **query 块**（大小 $B_q$）。上标 $(j)$：当前 **key 块**（大小 $B_k$）。

本作业测试里 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 为单头，形状 `(batch, seq, d)`。

| 符号 | 含义 | 形状（本作业） |
|------|------|----------------|
| $\mathbf{Q}$ | 查询 | $(B,\, N_q,\, d)$ |
| $\mathbf{K}$ | 键 | $(B,\, N_k,\, d)$ |
| $\mathbf{V}$ | 值 | $(B,\, N_k,\, d)$ |
| $\mathbf{S}$ | 缩放后的注意力分数 | $(B,\, N_q,\, N_k)$ |
| $\mathbf{P}$ | softmax 后的注意力权重 | $(B,\, N_q,\, N_k)$ |
| $\mathbf{O}$ | 注意力输出 | $(B,\, N_q,\, d)$ |
| $\mathbf{L}$ | logsumexp（**大写** $L$） | $(B,\, N_q)$ |
| $\mathbf{Q}_i$ | 第 $i$ 个 query 块 | $B_q \times d$ |
| $\mathbf{K}^{(j)},\,\mathbf{V}^{(j)}$ | 第 $j$ 个 key/value 块 | $B_k \times d$ |
| $\mathbf{S}_i^{(j)}$ | 第 $i$ 个 query 块与第 $j$ 个 key 块的分数子块 | $B_q \times B_k$ |
| $\tilde{\mathbf{P}}_i^{(j)}$ | 未归一化的 softmax 分子块 | $B_q \times B_k$ |
| $\mathbf{m}_i^{(j)}$ | 行方向运行最大值 | $B_q$ |
| $\mathbf{l}_i^{(j)}$ | softmax 分母的运行累加（**小写** $l$） | $B_q$ |
| $\mathbf{O}_i^{(j)}$ | 处理完 key 块 $1,\ldots,j$ 后的部分输出累加 | $B_q \times d$ |
| $B_q,\, B_k$ | tile 大小 | 各 $\geq 16$ |
| $T_q$ | query 块个数 | $\lceil N_q / B_q \rceil$ |
| $T_k$ | key 块个数 | $\lceil N_k / B_k \rceil$ |
| $d$ | **隐藏维**（$\mathbf{Q},\mathbf{K},\mathbf{V}$ 的最后一维） | 如 $64$ |

### $d$、$l$、$L$ 三个字母别混

| 符号 | 是什么 | 出现在哪 |
|------|--------|----------|
| **$d$** | 向量维度：$\mathbf{Q},\mathbf{K},\mathbf{V} \in \mathbb{R}^{N \times d}$；缩放用 $\sqrt{d}$ | 输入形状、$\mathbf{S}_i^{(j)} = \mathbf{Q}_i(\mathbf{K}^{(j)})^{\mathsf{T}}/\sqrt{d}$ |
| **$l_i^{(j)}$**（小写） | 内层循环里的 **分母累加器**，尚无 $\log$ | Algorithm 1 第 12 步 |
| **$L_i$**（大写） | 整行分数的 **logsumexp**，算法最后写出 | Algorithm 1 第 16 步：$L_i = m_i^{(T_k)} + \log(l_i^{(T_k)})$ |

**$d$ 与 $l$、$L$ 完全无关。** $d$ 是特征维；$l$ 是 softmax 分母在循环里的运行值；$L$ 是扫完所有 key 块后写进 HBM 的 logsumexp。

代码里：`d = q.shape[-1]` 是隐藏维；`l_i` 变量对应 $\mathbf{l}_i^{(j)}$；`l_out` 对应最终 $\mathbf{L}$。

---

## 二、前向：从朴素注意力到 Algorithm 1

### 2.1 朴素前向三步

$$
\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d}}
$$

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_{k} \exp(S_{ik})}
$$

$$
\mathbf{O} = \mathbf{P}\mathbf{V}
$$

数据流：$\mathbf{Q},\mathbf{K} \to \mathbf{S} \to \mathbf{P} \to \mathbf{O}$。

### 2.2 朴素前向的显存问题

$\mathbf{S}$ 与 $\mathbf{P}$ 形状都是 $(N_q,\, N_k)$。序列变长时，若把整张 $\mathbf{S}$、$\mathbf{P}$ 写入 HBM，显存按 **序列长度的平方** 增长。

FlashAttention 前向的目标：**$\mathbf{S}$、$\mathbf{P}$ 整块留在片上逐块算**；写回 HBM 的包括 $\mathbf{O}$，以及 Algorithm 1 要求的 $\mathbf{L}$（还有测试要求的 `save_for_backward` 里的 $\mathbf{Q},\mathbf{K},\mathbf{V}$）。

### 2.3 固定一行：softmax 在算什么

盯住查询行 $i$，键 $j = 0,\ldots,N_k-1$。

**分数：**

$$
S_{ij} = \frac{\mathbf{Q}_{i,:}\,\mathbf{K}_{j,:}^{\mathsf{T}}}{\sqrt{d}}
$$

**权重：**

$$
P_{ij} = \frac{\exp(S_{ij})}{\sum_{k} \exp(S_{ik})}
$$

分子是 $\exp(S_{ij})$；分母是 **整行** 所有 $\exp(S_{ik})$ 之和。四个键就是四格各自 $\exp$ 再相加。

**logsumexp（大写 $L_i$，Algorithm 1 第 16 步写出）：**

$$
L_i = \log \sum_{j} \exp(S_{ij})
$$

对「每个格子 $\exp$ 再相加」的结果取 $\log$。PyTorch：`torch.logsumexp(S, dim=-1)`。

**输出：**

$$
O_{i,:} = \sum_{j} P_{ij}\,\mathbf{V}_{j,:}
$$

一行视角：$S_{i,:} \to P_{i,:} \to O_{i,:}$。$L_i$ 与 $O_{i,:}$ 来自 **同一行** $\mathbf{S}$。

### 2.4 为何要在线维护 $\mathbf{m}$、$\mathbf{l}$、$\mathbf{O}_i^{(j)}$

朴素做法要对 **整行** $S_{i0},\ldots,S_{i,N_k-1}$ 一起做 softmax，再乘 $\mathbf{V}$，同时算 $L_i$。键很长时，整行 $\mathbf{S}$ 无法整块放进片上。

Algorithm 1 把键切成 $T_k$ 块，内层 $j=1,\ldots,T_k$，每次只加载 $\mathbf{K}^{(j)},\mathbf{V}^{(j)}$，算子块

$$
\mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i (\mathbf{K}^{(j)})^{\mathsf{T}}}{\sqrt{d}}
\in \mathbb{R}^{B_q \times B_k}
$$

用三个运行量把各块结果 **合并成与朴素前向相同的** $\mathbf{O}_i$ 和 $L_i$：

| 运行量 | Algorithm 1 | 作用 |
|--------|-------------|------|
| $\mathbf{m}_i^{(j)}$ | 第 10 步 | 截至块 $j$ 的行最大分数（数值稳定） |
| $\mathbf{l}_i^{(j)}$ | 第 12 步 | 截至块 $j$ 的分母累加：$\exp(m^{(j-1)}-m^{(j)})l^{(j-1)} + \operatorname{rowsum}(\tilde{\mathbf{P}}_i^{(j)})$ |
| $\mathbf{O}_i^{(j)}$ | 第 13 步 | 截至块 $j$ 的加权和累加（尚待第 15 步除以 $l_i^{(T_k)}$） |

其中第 11 步：

$$
\tilde{\mathbf{P}}_i^{(j)} = \exp\!\left(\mathbf{S}_i^{(j)} - \mathbf{m}_i^{(j)}\right)
$$

块 $1$ 单独算出的 softmax 只是局部信息；后续块可能带来更大 $S$，会更新 $m$、重标定 $l$ 和 $\mathbf{O}_i^{(j)}$。**必须扫完 $j=1,\ldots,T_k$。**

### 2.5 内层结束：写出 $\mathbf{O}_i$ 和 $\mathbf{L}_i$（Algorithm 1 第 15–16 步）

$$
\mathbf{O}_i = \operatorname{diag}\!\left((\mathbf{l}_i^{(T_k)})^{-1}\right) \mathbf{O}_i^{(T_k)}
$$

$$
\mathbf{L}_i = \mathbf{m}_i^{(T_k)} + \log\!\left(\mathbf{l}_i^{(T_k)}\right)
$$

第二式即 logsumexp。记 $m = m_i^{(T_k)}$，$l = l_i^{(T_k)}$，则 $l = \sum_j \exp(S_{ij}-m)$，故

$$
m + \log l = \log\sum_j \exp(S_{ij}) = L_i
$$

第一式用最终 $\mathbf{l}_i^{(T_k)}$ 归一化 $\mathbf{O}_i^{(T_k)}$，得到与 $\mathbf{P}\mathbf{V}$ 相同的 $\mathbf{O}_i$。

**前向写入 HBM：** $\mathbf{O}$、$\mathbf{L}$（及 `save_for_backward` 所需的 $\mathbf{Q},\mathbf{K},\mathbf{V}$）。

---

## 三、Algorithm 1（与任务描述逐步对应）

**Require:** $\mathbf{Q} \in \mathbb{R}^{N_q \times d}$，$\mathbf{K},\mathbf{V} \in \mathbb{R}^{N_k \times d}$，tile sizes $B_q,\, B_k$。

**Split:**

$$
T_q = \left\lceil \frac{N_q}{B_q} \right\rceil,
\qquad
T_k = \left\lceil \frac{N_k}{B_k} \right\rceil
$$

$\mathbf{Q}$ 切成 $\mathbf{Q}_1,\ldots,\mathbf{Q}_{T_q}$（各 $B_q \times d$）；$\mathbf{K},\mathbf{V}$ 切成 $\mathbf{K}^{(1)},\ldots,\mathbf{K}^{(T_k)}$ 与 $\mathbf{V}^{(1)},\ldots,\mathbf{V}^{(T_k)}$（各 $B_k \times d$）。

**for** $i = 1,\ldots,T_q$ **do**

1. Load $\mathbf{Q}_i$ from global memory

2. Initialize:

$$
\mathbf{O}_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q \times d},
\qquad
\mathbf{l}_i^{(0)} = \mathbf{0} \in \mathbb{R}^{B_q},
\qquad
\mathbf{m}_i^{(0)} = -\infty \in \mathbb{R}^{B_q}
$$

**for** $j = 1,\ldots,T_k$ **do**

3. Load $\mathbf{K}^{(j)},\,\mathbf{V}^{(j)}$ from global memory

4. Compute tile of pre-softmax attention scores:

$$
\mathbf{S}_i^{(j)} = \frac{\mathbf{Q}_i (\mathbf{K}^{(j)})^{\mathsf{T}}}{\sqrt{d}}
\in \mathbb{R}^{B_q \times B_k}
$$

5. Compute:

$$
\mathbf{m}_i^{(j)} = \max\!\left(\mathbf{m}_i^{(j-1)},\; \operatorname{rowmax}(\mathbf{S}_i^{(j)})\right)
\in \mathbb{R}^{B_q}
$$

6. Compute:

$$
\tilde{\mathbf{P}}_i^{(j)} = \exp\!\left(\mathbf{S}_i^{(j)} - \mathbf{m}_i^{(j)}\right)
\in \mathbb{R}^{B_q \times B_k}
$$

7. Compute:

$$
\mathbf{l}_i^{(j)} = \exp\!\left(\mathbf{m}_i^{(j-1)} - \mathbf{m}_i^{(j)}\right) \mathbf{l}_i^{(j-1)}
+ \operatorname{rowsum}\!\left(\tilde{\mathbf{P}}_i^{(j)}\right)
\in \mathbb{R}^{B_q}
$$

8. Compute:

$$
\mathbf{O}_i^{(j)} = \operatorname{diag}\!\left(\exp\!\left(\mathbf{m}_i^{(j-1)} - \mathbf{m}_i^{(j)}\right)\right) \mathbf{O}_i^{(j-1)}
+ \tilde{\mathbf{P}}_i^{(j)} \mathbf{V}^{(j)}
$$

**end for**

9. Compute:

$$
\mathbf{O}_i = \operatorname{diag}\!\left((\mathbf{l}_i^{(T_k)})^{-1}\right) \mathbf{O}_i^{(T_k)}
$$

10. Compute:

$$
\mathbf{L}_i = \mathbf{m}_i^{(T_k)} + \log\!\left(\mathbf{l}_i^{(T_k)}\right)
$$

11. Write $\mathbf{O}_i$ to global memory as the $i$-th tile of $\mathbf{O}$

12. Write $\mathbf{L}_i$ to global memory as the $i$-th tile of $\mathbf{L}$

**end for**

**Return** $\mathbf{O}$ and $\mathbf{L}$.

**$\operatorname{diag}(\mathbf{a})\,\mathbf{X}$：** 把 $\mathbf{X}$ 每行乘以 $\mathbf{a}$ 的对应元素。PyTorch：`torch.exp(a).unsqueeze(-1) * X`。

---

## 四、手算小例子

$N_q = N_k = 4$，$d = 2$，$B_q = 2$，$B_k = 2$。

$$
\mathbf{Q} =
\begin{pmatrix}
1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 2 & 0
\end{pmatrix},
\quad
\mathbf{K} =
\begin{pmatrix}
1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 2
\end{pmatrix},
\quad
\mathbf{V} =
\begin{pmatrix}
10 & 100 \\ 20 & 200 \\ 30 & 300 \\ 40 & 400
\end{pmatrix}
$$

只看 $i=1$（$\mathbf{Q}_1$ 为前两行）。

**初始化：** $\mathbf{O}_i^{(0)}=\mathbf{0}$，$\mathbf{l}_i^{(0)}=\mathbf{0}$，$\mathbf{m}_i^{(0)}=-\infty$。

**$j=1$：** $\mathbf{S}_i^{(1)} = \mathbf{Q}_1(\mathbf{K}^{(1)})^{\mathsf{T}}/\sqrt{2}$。全局行 $0$：$S_{0,0}\approx 0.707$，$S_{0,1}=0$；得 $m_i^{(1)}[0]=0.707$，$l_i^{(1)}[0]=1.493$，$\mathbf{O}_i^{(1)}[0]\approx [19.86,\,198.6]$。

**$j=2$：** 全局行 $0$ 最大仍为 $0.707$；$l_i^{(2)}[0]=2.986$。

**写回：**

$$
\mathbf{O}[0] = \mathbf{O}_i^{(2)}[0] / l_i^{(2)}[0],
\qquad
L[0] = m_i^{(2)}[0] + \log l_i^{(2)}[0]
$$

朴素验算 $L[0]=\log(e^{0.707}+1+e^{0.707}+1)\approx 1.31$，与 $m+\log l$ 相同。

---

## 五、$\mathbf{L}$ 与 $\mathbf{l}_i^{(j)}$

### 5.1 大写 $L_i$

$$
L_i = \log \sum_{j} \exp(S_{ij})
$$

Algorithm 1 第 16 步：$\mathbf{L}_i = \mathbf{m}_i^{(T_k)} + \log(\mathbf{l}_i^{(T_k)})$。测试用 `torch.logsumexp` 对朴素 $\mathbf{S}$ 验算。

### 5.2 小写 $l_i^{(j)}$

内层循环第 12 步的 **分母累加器**，每个 key 块更新一次，**尚无 $\log$**。扫完 $T_k$ 块后，与 $m_i^{(T_k)}$ 一起得到 $L_i$。

| | $L_i$（大写） | $l_i^{(j)}$（小写） |
|--|---------------|---------------------|
| 何时 | $j=T_k$ 之后 | 每个 $j$ 更新 |
| $\log$ | 有 | 无 |
| 写出 HBM | 是 | 否（片上） |

### 5.3 为何 $L_i$ 在内层循环结束后才算

只有 $l_i^{(T_k)}$、$m_i^{(T_k)}$ 包含 **全部** key 块的信息，此时

$$
L_i = m_i^{(T_k)} + \log(l_i^{(T_k)})
$$

才等于整行的 logsumexp。$\mathbf{L}$ 与 $\mathbf{O}$ 在同一次外层 $i$ 循环末尾一起写出。

---

## 六、任务 A / B / C

| 任务 | 内容 | 测试 |
|------|------|------|
| **A** | 纯 PyTorch 实现 Algorithm 1；`backward` → `NotImplementedError`；`is_causal=False` | `pytest -k test_flash_forward_pass_pytorch` |
| **B** | 内层 `for j` 写成 Triton `flash_fwd_kernel`；grid $(T_q,\,\text{batch})$ | `pytest -k test_flash_forward_pass_triton` |
| **C** | `is_causal`；mask 处 $\mathbf{S}_i^{(j)} \mathrel{+}= -10^{6}$；`ctx.is_causal = is_causal` | triton 测试含 causal |

任务 A 做对后，用 B 逐 op 对照 A。

---

## 七、任务 A：代码

### 7.1 文件

```
cs336_systems/flash_attention/flash_attn_pytorch.py
tests/adapters.py
```

### 7.2 朴素对照（验算 $\mathbf{O}$、$\mathbf{L}$）

```python
import math
import torch
from einops import einsum


def attention_reference(q, k, v):
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    s = einsum(q, k, "... q d, ... k d -> ... q k") * scale
    p = torch.softmax(s, dim=-1)
    o = einsum(p, v, "... q k, ... k d -> ... q d")
    l = torch.logsumexp(s, dim=-1)
    return o, l
```

### 7.3 Algorithm 1 的 PyTorch 实现

`o_i, m_i, l_i` → $\mathbf{O}_i^{(j)},\,\mathbf{m}_i^{(j)},\,\mathbf{l}_i^{(j)}$；`l_out` → $\mathbf{L}$。

```python
import math
import torch


def flash_attention_forward_pytorch(q, k, v, bq=16, bk=16):
    b, nq, d = q.shape
    nk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)

    o = torch.empty_like(q)
    l_out = torch.empty(b, nq, device=q.device, dtype=torch.float32)

    for i in range(0, nq, bq):
        q_i = q[:, i : i + bq, :]

        o_i = torch.zeros(b, bq, d, device=q.device, dtype=torch.float32)
        l_i = torch.zeros(b, bq, device=q.device, dtype=torch.float32)
        m_i = torch.full((b, bq), float("-inf"), device=q.device, dtype=torch.float32)

        for j in range(0, nk, bk):
            k_j = k[:, j : j + bk, :]
            v_j = v[:, j : j + bk, :]

            s_i_j = torch.matmul(q_i, k_j.transpose(-2, -1)) * scale

            m_i_prev = m_i
            m_i = torch.maximum(m_i, s_i_j.amax(dim=-1))

            p_tilde = torch.exp(s_i_j - m_i.unsqueeze(-1))

            l_i = torch.exp(m_i_prev - m_i) * l_i + p_tilde.sum(dim=-1)

            o_i = torch.exp(m_i_prev.unsqueeze(-1) - m_i.unsqueeze(-1)) * o_i
            o_i = o_i + torch.matmul(p_tilde, v_j)

        o[:, i : i + bq, :] = (o_i / l_i.unsqueeze(-1)).to(q.dtype)
        l_out[:, i : i + bq] = m_i + torch.log(l_i)

    return o, l_out
```

### 7.4 `autograd.Function`

```python
class FlashAttention2PyTorchFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        o, l = flash_attention_forward_pytorch(q, k, v, bq=16, bk=16)
        ctx.save_for_backward(l, q, k, v, o)
        return o

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError
```

### 7.5 `adapters.py`

```python
from cs336_systems.flash_attention.flash_attn_pytorch import FlashAttention2PyTorchFunc

def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttention2PyTorchFunc
```

### 7.6 测试

1. `q,k,v` 形状 $(4,\,128,\,64)$
2. `.apply(q, k, v, False)`
3. `saved_tensors` 里形状 $(4,\,128)$ 的张量有且仅有一个 → $\mathbf{L}$
4. `o`、`L` 与 `attention_reference` 比较，容差 $10^{-2}$

---

## 八、任务 B、C（简述）

**B：** 把内层 `for j` 换成 Triton；每个 program 一个 query tile × 一个 batch；片上 $\mathbf{O}_i^{(j)},\,\mathbf{l}_i^{(j)},\,\mathbf{m}_i^{(j)}$ 用 `tl.float32`。

**C：** `is_causal=True` 时对 mask 位置 $\mathbf{S}_i^{(j)} \mathrel{+}= -10^{6}$。

任务 A 的本质：PyTorch 双重循环实现 Algorithm 1，得到与朴素前向相同的 $\mathbf{O}$，并写出 $\mathbf{L}_i = m_i^{(T_k)} + \log(l_i^{(T_k)})$。
