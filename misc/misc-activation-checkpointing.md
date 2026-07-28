# Activation Checkpointing：讲义 §3.2 在说什么

上一节（§1.1 Operator Fusion）用 RMSNorm 说明：**算子融合**能减少「为反向保存的中间张量」。本节问一个更狠的问题：**就算把一个 `TransformerBlock` 用 `torch.compile` 融到极限，单层前向仍要囤多少保存张量？囤不下怎么办？**

答案的前半段是实测数字 **≈3651 MiB/层**；后半段是 **Activation Checkpointing（激活检查点 / 梯度检查点）**：少存、多算。

---

## 1. 讲义实验在干什么？

### 1.1 设定

```python
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(
    d_model=d_model, d_ff=d_ff, num_heads=num_heads,
    positional_encoder=RotaryEmbedding(
        dim=d_model // num_heads, context_length=context_length
    ),
)
block = torch.compile(block, fullgraph=True)   # 尽量融合
x = torch.randn((4, context_length, d_model), requires_grad=True)
```

- 输入 $x$：形状 `[4, 2048, 2560]`，即 $B=4,\ S=2048,\ d=2560$。
- **只跑一个** `TransformerBlock` 的前向（还没接 32 层整网）。
- 用 `saved_tensors_hooks` 的 `pack_hook` 累加 **所有被 autograd 登记为「要留到反向」的张量** 的字节数（跳过 `nn.Parameter`，避免把参数算两遍）。

### 1.2 结果

```
Total size of saved tensors in single TransformerBlock: 3651.31 MiB
```

**≈3.6 GiB** —— 仅仅 **一层** block、**一次**前向，autograd 就要为反向准备好约 3.6 GiB 的保存张量（讲义里叫 *residuals*；下文叫 **保存张量**，与 residual stream 无关）。

这和 memory profiling (f) 里量的 $R$（单层约 1.5 GiB，$S=512$）是 **同一类东西**，只是讲义这里 $S=2048$、且 block 已 `torch.compile`，数字更大：

- $S$ 从 512 涨到 2048 → attention 的 $S\times S$ 项按 $S^2$ 涨（约 16 倍），FFN 与 $(B,S,d)$ 项按 $S$ 线性涨。
- 所以 **fusion 省的是「同一层里重复、细碎」的保存**；**挡不住** 随 $B,S,d$ 变长而整体变大的激活。

讲义原话：*even with this fix, the memory use will grow linearly with batch size, sequence length and embedding size* —— 即 **对 $B$、$S$、$d$ 线性（或含 $S^2$ 项时更快）增长**。

### 1.3 Attention 里还有浪费

讲义提到 attention 的保存张量里仍有 **nontrivial waste**，Section 4 会用 FlashAttention 等办法再砍。本节 **先不管** 那部分，只讨论 checkpointing。

---

## 2. 背景：没有 checkpoint 时，显存为什么爆？

完整训练步的时间顺序（复习 memory profiling）：

1. **整网前向**：第 1 层算完 → 该层的保存张量 $R_1$ 占着显存；第 2 层 → $R_2$ 叠上去……第 32 层算完，$R_1+\cdots+R_{32}$ **同时活着**。
2. **反向**：从第 32 层往回，逐层读 $R_i$、写梯度、释放 $R_i$。

因此峰值显存 ≈ **参数 + 优化器 + 所有层的 $R$ 叠加**。单层 fused block 就 ~3.6 GiB（$S=2048$），32 层粗算就是 **百 GiB 量级** —— 即使用 fusion 也 OOM。

**核心矛盾：** 前向每算一层，就要 **一直留着** 这一层的中间结果，直到反向扫到这一层才能扔。层数 × 每层保存量 = _activation 显存的主项。

---

## 3. 思路：Recomputation（重算）代替「全存」

**Activation checkpointing** 的做法：

> 不在前向里保存 **每一个** 中间激活；只在若干 **检查点（checkpoint）** 保存 **进入某段计算的输入**；反向需要中间值时， **用检查点输入重新跑一遍那段前向**，临时算出来再用，用完即扔。

**代价：** 多算一次（或多次）前向 → **算力换显存**。  
**收益：** 任意时刻活着的保存张量，从「整段里所有层、所有算子的 $R$」变成「只有检查点边界上的几个输入」。

---

## 4. `torch.utils.checkpoint.checkpoint` 具体改了什么？

签名概念：`checkpoint(fn, *args, use_reentrant=False)`  
把 `fn` 包成「带检查点语义」的函数。

### 4.1 前向（Forward）

对包在 `checkpoint` 里的 `fn`：

1. **保存** `fn` 的 **输入**（例如进入 `two_blocks` 的 $x$，形状 `[4,2048,2560]`）。
2. **压制** `fn` **内部** 前向的 saved-tensor 登记 —— `pack_hook` 在 `fn` 执行期间 **几乎看不到** 层内的 80 MiB、128 MiB 等保存项。
3. `fn` 的 **输出** 照常往前传（计算图仍连着，loss 仍能反传）。

直觉：前向穿过 checkpoint 区域时，显存里 **只留下「进区域的门票」$x$**，区域内的 FFN/Attention 中间激活 **不落盘给 autograd**。

### 4.2 反向（Backward）

当 autograd 反传到这个 checkpoint 节点时：

1. **先重算（recompute）**：用前向存下的输入， **再跑一遍** `fn` 的前向；这一次 **允许** 正常登记保存张量（供紧接着的 backward 用）。
2. **再反传**：对 `fn` 内部按普通规则 backward；用完后这些临时保存张量 **释放**。

直觉：反向扫到检查点时， **临时** 把区域内前向「补跑」一次，换出中间激活，算完梯度就扔掉 —— 不需要从 **整网前向结束** 一直囤到 **整网反向开始**。

### 4.3 `use_reentrant=False`

讲义使用 `use_reentrant=False`（PyTorch 2 推荐）。与旧版 `reentrant=True` 相比，语义更清晰、与 `compile` / `saved_tensors_hooks` 配合更稳。本讲义实验按此写法即可。

---

## 5. 讲义例子：4 个 block，有 / 无 checkpoint

假设同一个已 `compile` 的 `block`。

### 5.1 无 checkpoint：`four_blocks`

```python
def four_blocks(x):
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks(x)
```

前向路径：

```
x ──► block ──► block ──► block ──► block ──► y
      R₁       R₂       R₃       R₄   （四层保存张量全程叠在显存里）
```

`pack_hook` 会登记 **4 层内部** 几乎全部保存张量 → `total_size_bytes` 约为 **4 × 单层 ~3651 MiB** 量级（具体略少/略多取决于图共享，但趋势是 **线性乘层数**）。

### 5.2 有 checkpoint：`four_blocks_checkpoint`

```python
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
```

结构：**2 个检查点**，每个检查点里 **2 个 block**。

前向路径（**穿过** checkpoint 时，区域内不囤 $R$）：

```
x ──► [checkpoint: two_blocks] ──► h ──► [checkpoint: two_blocks] ──► y
      只存进入时的 x              只存进入时的 h
      内部 2×block 的 R 不登记
```

`pack_hook` 在 **整段** `four_blocks_checkpoint` 前向里看到的 Saving 大致是：

```
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32,
    grad_fn=<torch.autograd.function.CompiledFunctionBackward ...>
```

逐行读：

| 行 | 形状 | 含义 |
|----|------|------|
| `[0]` | 空张量占位 | checkpoint / `compile` 内部 bookkeeping，**0 字节**，可忽略 |
| `[4, 2048, 2560]` | 第一个 `checkpoint(two_blocks, x)` 的 **输入** $x$ | 进入「block1+block2」区域的门票；$4\cdot2048\cdot2560\cdot4/1024^2 =$ **80 MiB** |
| `[0]` | 同上 | 第二个 checkpoint 的占位 |
| `[4, 2048, 2560]` | 第二个 `checkpoint(two_blocks, h)` 的 **输入** $h$ | 第一个 `two_blocks` 的输出，作为第二段的输入；再 **80 MiB** |

**整段前向** `pack_hook` 累加 ≈ **160 MiB**（两张 residual-stream 形状的输入），而不是无 checkpoint 时 **~4 × 3651 MiB**。

第二行带 `CompiledFunctionBackward`：说明第二个 checkpoint 的输入挂在 **编译后的自定义反传节点** 上，与 §1.1 fusion 一致 —— 检查点边界仍是一个「粗粒度」图节点。

### 5.3 反向时时间线（第二个 checkpoint 为例）

```
1. 反传信号传到第二个 checkpoint 的出口
2. checkpoint 用保存的 h 重新执行 two_blocks 前向 → 临时产生 block3、block4 的保存张量
3. 对 two_blocks 做 backward → 释放这些临时张量
4. 把梯度传给 h，继续往前
5. 第一个 checkpoint 同理，用保存的 x 重算 block1+block2，再 backward
```

**任意时刻**：最多 **「当前正在 backward 的那一段 two_blocks」** 的保存张量活着，而不是 4 段叠在一起。

---

## 6. 和 fusion、memory profiling 怎么串起来？

| 手段 | 解决什么 | 不解决什么 |
|------|----------|------------|
| **Operator fusion**（§1.1） | 单层 **内部** 少存重复、细碎 saved tensor（如 RMSNorm 多份 20 MiB） | 层数 × 每层仍随 $B,S,d$ 变大；32 层仍叠 $R$ |
| **Fused TransformerBlock**（§3.2 实测） | 单层内 kernel 更少、保存列表更短 | 单层仍 ~3.6 GiB（$S=2048$） |
| **Activation checkpointing**（§3.2.1） | **跨层** 不同时囤所有 $R$；只留检查点输入 | 反向要多算前向；算力 ↑ |

Memory profiling 报告里：无 checkpoint 时整网前向峰值高、反向台阶下降 —— 正是因为 **32 层 $R$ 叠满**。若在模型里对每 $k$ 层包一层 `checkpoint`，峰值会从「$\propto L$ 层保存」降为「$\propto$ 检查点段长 + 重算临时量」，用 **额外前向计算** 换 **可训练的 batch/context**。

---

## 7. 检查点怎么切？（讲义 4 block 的启示）

- 4 层切成 **2×2**：2 个 checkpoint，前向只存 **2 张** `[B,S,d]` 输入（≈160 MiB）。
- 32 层整网常见做法：每 **1 层** 或每 **2～4 层** 一个 checkpoint，在「存多少」和「重算几次」之间折中。
- 段越长 → 前向存的边界越少，但 **每次 backward 重算的前向越长**（算力惩罚越大）。

---

## 8. 一句话总结

> **Fusion 让单层「存得更省」；checkpointing 让多层「不要同时都存」。**  
> `checkpoint(fn, x)` 在前向只保留进入 `fn` 的 $x$，压制 `fn` 内部的保存张量；反向时再跑一遍 `fn` 的前向换出中间激活，算完即释。讲义 4-block 例子把保存量从「四层各 ~3.6 GiB 叠加」收成「两个 checkpoint 边界各一张 80 MiB 的 `[4,2048,2560]`」——这就是 activation checkpointing 的核心交易：**多算一次前向，少占一份显存。**

---

## 附录：与 operator fusion 文档的关系

- `misc/misc-operator-fusion-rmsnorm.md`：单层 **算子粒度** 与 saved tensor 列表。
- 本文：多层 **时间维度** 上何时保留 / 何时重算 saved tensor。
- 二者常 **一起用**：`torch.compile(block)` + `checkpoint(every_n_layers, ...)` 是工业界训练大模型的常规组合。
