# Gradient Checkpointing 作业题：认知地图

> 对应 Problem **gradient_checkpointing** (a)(b)。与 `misc-activation-checkpointing.md`（讲义 §3.2 机制）衔接：那边讲 **checkpoint API 怎么工作**；本文讲 **这题怎么答、怎么测**。

---

## 0. 题设在说什么

$N$ 个 **相同** `TransformerBlock` 顺序堆叠（xl：$N=32$）。每个 block 前向会为反向留下保存张量，体积记为 $R$（单层主导项；$S=2048,\,B=4$ 时约 $3.5\text{–}4\,\mathrm{GiB}$，见 `measure_block_saved_tensors.py`）。

| 模式 | 前向结束时活着的 block 级保存量 | 峰值激活（主导项） |
|------|--------------------------------|-------------------|
| 无 checkpoint | $N$ 层各自的 $R$ **同时**在显存里 | $\Theta(N \cdot R)$ |
| 有 checkpoint | 只留 **检查点边界** 的输入；层内 $R$ 前向不囤，反向再算 | 由 **边界数量** 与 **重算段长** 共同决定 |

作业允许把 forward 的 **任意子段** 包进 `checkpoint`，且 **可嵌套**（(a)）；(b) 则 **禁止嵌套**，且只有 **一层** 重算语义。

---

## 1. 一个 checkpoint 段在干什么

包一段函数 `fn`（例如连续 $k$ 个 block）：

**前向**

1. 保存 `fn` 的 **输入** $x$（形状 $(B,S,d)$，xl·$S{=}2048$ 时约 $80\,\mathrm{MiB}$）。
2. **不登记** `fn` 内部的保存张量（各层 $R$ 不落盘）。

**反向**（扫到该段时）

1. 用存下的 $x$ **再跑一遍** `fn` 的前向 → 临时产生段内 $R$。
2. 对 `fn` 正常 backward → 释放临时 $R$。

一段 checkpoint 在 backward 里增加 **一次** 该段前向的计算；**(b) 的「one step of recomputation」= 每个 checkpoint 区间只重算这一遍，不再在重算里再套 checkpoint。**

---

## 2. Part (a)：不算算力代价时的最优策略

### 2.1 目标

$$\min \;\text{peak activation memory} \qquad \text{（compute 无限便宜）}$$

### 2.2 策略

**每个 block 单独包一层 `checkpoint`**（$k=1$，不嵌套也已够；嵌套不会比「每层一段」更省显存，只会多此一举）。

```python
def forward_checkpoint_every_layer(x, layers):
    for block in layers:
        x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
    return x
```

任意时刻：反向正在重算的 block **至多一个** → 同时存活的 block 级 $R$ 为 **$O(1)$** 份。  
另还有 $N$ 个 checkpoint 边界输入，各 $(B,S,d)$；题设 **单层 $R$ 主导 bookkeeping**，故

$$\boxed{\text{peak activation} = \Theta(R_{\text{block}}) = O(1)\ \text{随}\ N}$$

（相对 $N$ 常数；随 $B,S,d$ 仍变。）

### 2.3 算力（(a) 也要写）

无 checkpoint：前向 $\Theta(N)$ 个 block。  
每层 checkpoint：反向每层 **多算 1 次** block 前向 → 额外 $\Theta(N)$ 次 block-forward，

$$\boxed{\text{总前向工作量} = \Theta(N)\ \text{（原）} + \Theta(N)\ \text{（重算）} = \Theta(N)}$$

（常数因子约 $\times 2$ 量级，渐近仍是线性。）

### 2.4 与 $\sqrt{N}$ 策略的区分

Chen 等「$\sqrt{N}$ 内存、$\sqrt{N}$ 额外前向」是在 **算力也约束** 时的折中。**(a) 明确 ignore compute**，答案不是 $\sqrt{N}$，而是 **切到最细（每层一段）**。

---

## 3. Part (b)：不嵌套 + 一层重算时怎么选段长 $k$

### 3.1 约束

- 只能 **一层** `checkpoint` 包装：把 $N$ 层切成 $\lceil N/k \rceil$ 段，每段 **连续 $k$ 个 block**。
- 不能在 `checkpoint` 里面再 `checkpoint`。

```python
def run_k_blocks(x, blocks, k):
    for block in blocks[:k]:
        x = block(x)
    return x

def forward_segments(x, layers, k):
    for i in range(0, len(layers), k):
        chunk = layers[i : i + k]
        x = torch.utils.checkpoint.checkpoint(
            lambda t, m=chunk: run_k_blocks(t, m),
            x,
            use_reentrant=False,
        )
    return x
```

（实现时用具名函数代替 `lambda`，避免闭包坑；此处仅示意。）

### 3.2 峰值粗算（两段竞争）

记边界张量大小 $B_{\mathrm{bd}} = B\cdot S\cdot d\cdot 4$（字节），单层保存量 $R$。

| 阶段 | 主导项 |
|------|--------|
| 前向结束 | $\dfrac{N}{k}\cdot B_{\mathrm{bd}}$（各段入口 $x$ 仍要留着） |
| 反向重算某段 | $k \cdot R$（段内 $k$ 层 $R$ 临时叠出） |

$$\text{peak}(k) \approx \max\!\left(\frac{N}{k}\,B_{\mathrm{bd}},\; k\,R\right)$$

- $k$ **大** → 边界少，但重算时 $kR$ 爆（$k{=}N$ 时反向要临时囤几乎整网 $R$）。
- $k$ **小** → $kR$ 小；$k{=}1$ 时重算项为 $R$，边界项为 $N\cdot B_{\mathrm{bd}}$。

在 $R \gg B_{\mathrm{bd}}$（单层 3.6 GiB vs 边界 80 MiB）时，**$kR$ 项主导** → **$k$ 越小越好**；$k=1$ 通常最优。  
$N=32,\,S=2048$ 粗算：$k{=}1$ 时 $\max(2.5\,\mathrm{GiB},\,3.6\,\mathrm{GiB})\approx 3.6\,\mathrm{GiB}$；$k{=}2$ 时 $\max(1.25,\,7.2)\approx 7.2\,\mathrm{GiB}$。→ **假设：最优 $k=1$**，须用 GPU **实测** 并比较 $k\in\{1,2,3\}$（或最优 $k$ 的 $k{-}1,k,k{+}1$）。

### 3.3 怎么测（与 memory profiling 同口径）

| 项 | 选择 |
|----|------|
| 模型 | xl，$B=4$，$S=2048$ |
| 步类型 | **完整训练步**：forward → loss → backward → `AdamW.step()` |
| 指标 | `torch.cuda.max_memory_allocated()`（该 step 全局峰值） |
| 对照 | 无 checkpoint；若干 $k$；最优 $k$ 邻域 $k{-}1,k,k{+}1$ |
| 代码 | **调用侧** 包 checkpoint；**不改** `cs336_basics/model.py`；复用 `e2e_timing` 建模型与 batch |

### 3.4 交付物对照

| 问 | 交付 |
|----|------|
| (a) | 策略 3–5 句 + 渐近 peak / compute + code sketch |
| (b) | 为何选该 $k$ 3–5 句 + **实测** peak + 相邻 $k$ 对比表 |

---

## 4. 实现时要动的几块（一览）

```
gradient_checkpointing/     ← 新建模块（或扩 memory_profiling）
  ├── forward_segments(layers, k)   # checkpoint 切块
  ├── profile_step(k | None)        # None = 无 checkpoint baseline
  └── sweep → peaks.json + report

复用：e2e.build_model, make_batch, xl preset
不测：saved_tensors_hooks 累加（那是 (a) 讲义玩具；本题 (b) 要 max_memory_allocated）
```

---

## 5. 读题检查清单

1. (a) 与 (b) **策略可能不同**：(a) 每层 checkpoint；(b) 在不能 nest 时仍可选 $k$，但 $R\gg B_{\mathrm{bd}}$ 时 $k{=}1$ 常最优。
2. **嵌套**：仅 (a) 可讨论；**(b) 禁止**。
3. **峰值**：整步 `max_memory_allocated`，不是 forward 时 hooks 加总。
4. **验证**：必须跑 **$k{-}1,k,k{+}1$**（在可行范围内），不能只报一个点。

---

## 6. 一句话

> **(a)**：算力不限 → 每层 `checkpoint` → peak $O(R_{\text{block}})$，extra compute $\Theta(N)$。  
> **(b)**：只能一层 checkpoint → 在段长 $k$ 上权衡 $\frac{N}{k}B_{\mathrm{bd}}$ 与 $kR$ → 理论倾向 $k{=}1$ → **profile 完整训练步** 证实并和 $k{=}2,3$ 比。
