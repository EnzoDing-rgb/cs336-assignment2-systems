# Operator Fusion 与 RMSNorm：讲义 §1.1 在说什么

讲义用 `saved_tensors_hooks` 对比 **融合前 / 融合后** 的 RMSNorm，表面上是打印行数变少了，背后其实是 **计算图粒度** 和 **反向传播要存什么** 两件事。本文按「问题 → 机制 → 融合改变了什么 → 深意」展开。

---

## 1. 讲义在抱怨什么？

讲义原话：**the granularity of the operations used is too high**（算子粒度太细）。

我们的 `RMSNorm.forward`（`cs336_basics/model.py`）在 PyTorch 里会被拆成很多个小算子，而不是「一个 RMSNorm」：

```python
x = x.to(torch.float32)                                    # 算子 1：dtype 转换
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # 算子 2–5：pow → mean → add → rsqrt
x = x * rms                                                # 算子 6：逐元乘
return (self.weight * x).to(in_dtype)                      # 算子 7–8：乘 γ → 转回 dtype
```

每一个小算子都会在 autograd 里注册成一个 **独立的节点**。前向每经过一个节点，反向就可能要 **单独保存** 该节点的输入（讲义里叫 *residual*；下文统一叫 **保存张量 / saved tensor**，避免和 residual stream 混淆）。

**粒度太细的后果：**

| 维度 | 细粒度（未融合） | 粗粒度（融合后） |
|------|------------------|------------------|
| 计算图 | 一条 RMSNorm ≈ 8 个小节点 | 一条 RMSNorm = **1 个** 自定义节点 |
| 保存张量 | 每个小节点各存自己需要的输入，可能 **重复存** 同形状的 $(B,S,d)$ | 整个 RMSNorm 只存 **一份** 协议约定好的最小集合 |
| 反向 | 按节点逐个 pop saved tensor，顺序 = 前向的逆序 | **一个** fused backward kernel 按需读取 |
| Kernel 启动 | 多次 GPU kernel，中间结果写回显存 | 理想情况下一次 kernel 算完，少写中间 buffer |

讲义要的理想形态：**一个 op 吃进去 `(γ, x)`，吐出来 `y`；反向也是「一个 op」**，前后对称、unitary。

---

## 2. `saved_tensors_hooks` 在量什么？

讲义代码骨架：

```python
def pack_hook(tensor):
    print("Saving residual:", tensor.shape, tensor.dtype, tensor.grad_fn)
    return tensor

def unpack_hook(tensor):
    print("Loading residual:", tensor.shape, tensor.dtype, tensor.grad_fn)
    return tensor

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

- **Saving**：前向时，autograd 决定「这个张量要留到反向」→ 调用 `pack_hook`。
- **Loading**：反向时，某个算子需要读保存张量 → 调用 `unpack_hook`。

所以它量的不是「cudaMalloc 了几次」，而是 **autograd 登记了多少份 saved tensor、各是什么形状**。这正是 memory profiling (f) 里 `saved_tensors_hooks` 的用法。

---

## 3. 融合前：为什么又乱又费？

未 `torch.compile` 时，典型现象（讲义「旧输出」，行数多、形状杂）：

- 会出现 **多张** 与 $(B,S,d)$ 同量级的大张量被保存（例如 `pow` 的输入、`mean` 的输入、两次 `mul` 的输入等），有的还带 **不同的 `grad_fn`**（指向不同子算子）。
- **Saving 顺序** 严格跟著前向微算子走；**Loading 顺序** 是 Saving 的 **逆序**（栈式 LIFO）——因为 autograd 从 loss 往回，先碰到最后注册的那个 saved tensor。

直觉图（未融合）：

```
前向（细）:  x → to_fp32 → pow → mean → rsqrt → *rms → *γ → y
              ↓save?  ↓save?  ↓save?  ↓save?   ↓save? ↓save?
反向（细）:  从 y 往回，每个小节点 pop 自己当初 save 的那份
```

每个小节点的 backward 只懂自己的数学（例如 `pow` 的导数），所以 PyTorch **不得不** 在前向把该节点的输入单独登记进 saved 列表。多个节点都需要「那个 $(4,512,2560)$ 的 x 或它的变体」，列表里就会出现 **多次** 大形状条目（有时是引用同一块显存，但 hooks 仍会 **按次计数**）。

---

## 4. 融合后：新输出三行分别是什么？

`ln = torch.compile(RMSNorm(...))` 之后，讲义给出的新输出：

```
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
```

这三行是 **整个 RMSNorm 作为单一自定义算子** 时，反向所约定的 **最小保存集**：

| 形状 | 是什么 | 为何反向需要 |
|------|--------|--------------|
| `[4, 512, 2560]` | RMSNorm 的 **输入** $x$ | 算 $\partial L/\partial x$ 必须知道前向喂进来的 $x$ |
| `[2560]` | 可学习缩放 **γ**（`self.weight`） | 算 $\partial L/\partial \gamma$ |
| `[4, 512, 1]` | 见下节 | 前向做归一化时已经算好的 **每个 token 一个数** 的缩放因子，反向要用、又不想重算整条 `pow → mean → rsqrt` |

体积粗算（$B{=}4,S{=}512,d{=}2560$，FP32）：

- 输入 $x$：$4\cdot512\cdot2560\cdot4/1024^2 \approx$ **20 MiB**
- γ：可忽略（$\sim$0.01 MiB）
- `[4, 512, 1]`：$4\cdot512\cdot1\cdot4/1024^2 \approx$ **0.008 MiB**（2048 个 float，每个 token 只占 4 bytes）

**关键对比：** 未融合时，hooks 可能登记 **多份** 20 MiB 级别的张量；融合后 **整张 $(B,S,d)$ 只登记一次**，中间 `pow` / `mean` / `rsqrt` 的大块不再各自占 saved 列表。

讲义原句 *We only need to save a single full-size activation tensor* —— 指的就是这一份输入 $x$。

### `[4, 512, 1]` 到底是什么？（对应讲义第三行 Saving）

先写 RMSNorm 在算什么。我们的实现（`cs336_basics/model.py`）是：

```python
x = x.to(torch.float32)                                    # x: (4, 512, 2560)
rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # rms: (4, 512, 1)  ← 就是它
x = x * rms                                                # 每个 token 用同一个数去乘 2560 维向量
return (self.weight * x).to(in_dtype)                      # 再乘 γ
```

对 **某一个** batch 位置 $b$、token 位置 $s$，在 hidden 维 $d=2560$ 上做：

$$
\text{rms}[b,s,0]
= \frac{1}{\sqrt{\dfrac{1}{d}\sum_{k=1}^{d} x[b,s,k]^2 + \varepsilon}}
= \texttt{rsqrt}\bigl(\texttt{mean}(x[b,s,:]^2) + \varepsilon\bigr).
$$

所以 **`[4, 512, 1]` 不是又一张「小号的激活图」**，而是：

- 形状 `(batch, seq, 1)`：最后一维是 `keepdim=True` 故意留的，方便和 `(B,S,d)` 做广播乘法；
- 每个格子 **只有一个 float**：这个 token 的 **归一化缩放因子**（把该 token 的 2560 维向量除以其 RMS 长度）；
- 它就是前向里变量 `rms` 的值——`torch.rsqrt(...)` 的 **直接输出**。

**前向怎么用：** `x * rms` 时，`rms[b,s,0]` 这一个数会乘上 $x[b,s,0], x[b,s,1], \ldots, x[b,s,2559]$ 全部 2560 个分量。

**反向为什么存它：** 输出是 $y = \gamma \odot (x \odot \text{rms})$（$\odot$ 为逐元乘，rms 沿 $d$ 广播）。对 $x$ 求导时，链式法则里会出现前向用过的 **那个** $\text{rms}[b,s,0]$（以及 $x$ 本身）。融合后的自定义 backward 选择 **把前向算好的 rms 存下来**（2048 个数，0.008 MiB），而不是：

- 再存一张归一化后的 $x$（又是 20 MiB），或
- 反向时重新跑一遍 `pow → mean → rsqrt`（多一次 kernel、多占临时显存）。

一句话：**`[4, 512, 1]` = 每个 token 一个的 `rsqrt(mean(x²)+ε)`，前向归一化乘上去的那个因子，反向求 $\partial L/\partial x$ 时要原样拿出来用。**

---

## 5. 两个「诡异」现象的含义

### 5.1 为什么 `grad_fn=None`？

未融合时，saved tensor 往往带 `grad_fn=<PowBackward0>` 之类——说明它仍挂在 **某个微算子** 的子图上。

融合后 `grad_fn=None`：**不是** 说张量可脱离计算图，而是说 PyTorch 把整个 RMSNorm 编译成了 **一个 CustomAutogradFunction**（或 Inductor 生成的等价物）。保存张量挂在 **这个融合算子** 上，不再逐个挂在 `pow` / `mean` 上，所以打印出来是 `None`（或统一的 fused backward 节点）。

含义：**autograd 眼里 RMSNorm 已经是一个黑盒 op**，不再暴露内部 8 步。

### 5.2 为什么 Loading 顺序不再与 Saving 相反？

未融合：反向 = 沿微算子链往回走 → saved tensor 栈 **后进先出**，Loading 顺序与 Saving **严格相反**。

融合后：**只有一个 backward 入口**。融合 kernel 里按算法需要读取 $(x,\ \gamma,\ \text{rms})$，不必遵守「8 个小算子各自 pop」的栈顺序。因此 Saving 顺序是 `[大, 小, 中]` 时，Loading 未必严格逆序——**这是预期行为**，说明 backward 已是 **单元算子**，不是微算子链的机械反转。

讲义 *each residual no longer has a grad_fn dependency* —— 依赖关系从「8 层微算子链」收成「1 个融合算子 ↔ 1 套 saved tensors」。

---

## 6. 「unitary in the backward pass」是什么意思？

讲义要求 forward / backward **粒度对称**：

- **Forward：** 一个 RMSNorm op：$(\gamma, x) \mapsto y$
- **Backward：** 一个 RMSNorm backward op：$(\partial L/\partial y,\ \text{saved}) \mapsto (\partial L/\partial x,\ \partial L/\partial \gamma)$

对 autograd 而言是 **一对一** 的自定义函数，而不是 8 个 forward 节点配 8 个 backward 节点、各自 save/load。

这也是 **kernel fusion** 和 **autograd fusion** 绑在一起的原因：`torch.compile` 不只把 CUDA kernel 拼快， often 还会生成 **融合的反向 kernel**，从根上减少 saved tensor 种类和 kernel launch 次数。

---

## 7. 和 memory profiling 怎么接上？

| 讲义 §1.1 | 你的 profiling 报告 |
|-----------|---------------------|
| 单算子 RMSNorm 的 saved 列表 | 整层 / 整网用 hooks 汇总成 $R$（如 FFN 680 MiB） |
| 粒度太细 → 保存张量多 | SwiGLU、Attention 未融合时，一层里 **多张** 80 MiB / 128 MiB |
| `torch.compile` 示范 | 同一思路可推广：compile 子模块 → 减少重复 saved activation |
| 只存 $x$ + $\gamma$ + 每个 token 的 rms 因子 | 融合目标：用 **一份 20 MiB 的 $x$** + **2048 个 float 的 rms** 换掉多份中间激活 |

RMSNorm 单层省下的显存不大（每层 2 次 RMSNorm，每次从「可能多份 20 MiB」收到「1 份 $x$ + 一份 `[B,S,1]` 的 rms」）。但 **Attention / FFN** 若也能融合，省的是 **80 MiB × 8.5 张** 这种量级——这才是 fusion 在训练大模型里真正值钱的地方。

---

## 8. 一句话总结

讲义 §1.1 的深意不是「`torch.compile` 会打印更少行」，而是：

> **算子粒度决定 autograd 要囤多少 saved tensor。把 RMSNorm 从 8 个微算子熔成 1 个自定义算子，反向只需保存「输入 $x$ + 参数 $\gamma$ + 每个 token 的 rms 缩放因子 `[B,S,1]`」，且 backward 本身也变成单一、可融合的内核——这就是 operator fusion 在显存与性能上的动机。**

`saved_tensors_hooks` 是肉眼可见的证物：`grad_fn` 消失、load 顺序不再机械反转、full-size 激活从「可能多份」变成「确定一份」。

---

## 附录：本地可复现（可选）

```python
import torch
from cs336_basics.model import RMSNorm

def pack_hook(t):
    print("Saving:", t.shape, t.dtype, t.grad_fn)
    return t

def unpack_hook(t):
    print("Loading:", t.shape, t.dtype, t.grad_fn)
    return t

x = torch.randn(4, 512, 2560, device="cuda", requires_grad=True)

# A) 未融合
ln = RMSNorm(2560, device="cuda")
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()

x = torch.randn(4, 512, 2560, device="cuda", requires_grad=True)

# B) torch.compile 融合（需 CUDA + PyTorch 2.x）
ln = torch.compile(RMSNorm(2560, device="cuda"))
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward()
```

对比 A/B 的 Saving 行数与形状，即讲义实验。
