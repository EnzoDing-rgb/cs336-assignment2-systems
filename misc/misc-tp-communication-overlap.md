# 推理并行：TP / CP / EP 与 Prefill–Decode 差异

> **范围：** 只谈 **推理**（inference），不谈 Pipeline Parallel（PP），也不谈训练里的 DDP/FSDP backward。  
> 推理里你最常碰到的三种并行：**TP**（张量并行）、**CP**（上下文/序列并行）、**EP**（专家并行）。  
> 本文先建立 **prefill vs decode** 的心智模型，再讲各并行「切什么、传什么、能不能 overlap」。

**相关文档：**

| 主题 | 文件 |
|------|------|
| TP 列切/行切、FFN 通信量 | [tensor-parallel-calculations.md §1–§2](../reports/tensor-parallel-calculations.md) |
| TP / CP / EP 对照表 | [misc-llm-parallelism-table.md §5–§7](./misc-llm-parallelism-table.md) |
| KV cache、逐步 forward | [misc-kv-cache-explained.md](./misc-kv-cache-explained.md) |
| AllReduce = RS + AG | [alternate-ring-all-reduce.md §1](../reports/alternate-ring-all-reduce.md) |
| FFN TP 可视化 | [misc-tp-ffn-simple.html](./misc-tp-ffn-simple.html) |

---

## 1. 推理的两段人生：Prefill 与 Decode

同一次生成请求，GPU 上其实是 **两种完全不同的算力形态**：

| | **Prefill（提示词阶段）** | **Decode（逐 token 生成）** |
|--|---------------------------|------------------------------|
| **输入** | 整段 prompt，长度 $S_{\mathrm{prompt}}$（可上千） | 每步 **1 个新 token**（连续 batching 时 $B_{\mathrm{eff}}>1$） |
| **Attention 在算什么** | 所有位置 **一次算完**（因果 mask） | 只算 **新 token 的 Q**；K/V 从 **KV cache 读** |
| **计算量级（单层 Attention）** | $\mathcal{O}(S^2)$（$QK^\top$ 是 $S\times S$） | $\mathcal{O}(S)$（一个 query 对所有历史 key） |
| **典型瓶颈** | **算力**（大 matmul、Tensor Core） | **访存 / 延迟**（小 matmul + 读很长 KV） |
| **KV cache** | **写入**：为每个 prompt 位置存 K/V | **追加** 1 行 K/V；其余 **只读** |
| **TP all-reduce 的激活形状** | $(B,\, S,\, D)$ — **$S$ 大** | $(B,\, 1,\, D)$ — **$S$ 小、但每层每 token 都来一次** |

```text
用户 prompt ──► [ Prefill: 一口气算完 S_prompt 个 token ] ──► 采样第 1 个 token
                                                      │
                                                      ▼
                                    [ Decode: 每步 +1 token，重复 L 层 × 每 token 一步 ]
                                                      │
                                                      └── 直到 EOS / 长度上限
```

**记忆：** Prefill 像「批处理大矩阵乘」；Decode 像「带巨大只读缓存的极窄矩阵乘，循环成千上万次」。

无 KV cache 的朴素 generate（见 [misc-kv-cache-explained.md](./misc-kv-cache-explained.md)）每步重算整段——那是教学用；**生产推理一定用 cache**，所以 **decode 的 $S$ 在 matmul 里体现为读 cache 长度，而不是当前 Q 的行数**。

---

## 2. 推理关心的三种并行：各切什么

不谈 PP/DP，只列推理栈里 **TP / CP / EP** 分工（详表见 [misc-llm-parallelism-table.md](./misc-llm-parallelism-table.md)）：

| 并行 | 切什么 | 主要降什么 | 主要通信 |
|------|--------|------------|----------|
| **TP** | 层内大矩阵（列/行切） | **权重** $\sim 1/N_{\mathrm{TP}}$ | 每层 block：**激活 all-reduce**（$\sim bsh$） |
| **CP** | **序列维** token / KV 分片 | **激活 + KV** $\sim 1/N_{\mathrm{CP}}$ | 层间 **序列分片交换**（凑齐 attention 所需 K/V） |
| **EP** | **MoE 专家**权重 | **专家参数** $\sim 1/N_{\mathrm{EP}}$ | 每层 MoE：**token all-to-all**（dispatch + combine） |

```text
                    ┌─ TP：把「宽矩阵」切开算
  一层 Transformer ─┼─ CP：把「长序列 / KV」切开存
                    └─ EP：把「多个专家 FFN」拆开放（仅 MoE 层）

  三者可叠：常见「节点内 TP + CP 撑长上下文 + EP 撑 MoE 参数量」
```

**纯 TP、不带 CP 时：** 每张 TP 卡 **复制整段序列** 的 hidden state；KV cache 也 **每卡一份完整长度**（或按 TP+SP 约定切分中间宽激活，见 TP 表注）。**长上下文 KV 爆显存 → 上 CP**，不是再加 TP。

---

## 3. TP 基础：FFN 哪步要通信

Megatron FFN（[tensor-parallel-calculations.md §1.3](../reports/tensor-parallel-calculations.md)）：

```text
x ──► W1,W2 列切 ──► x1,x2 ──► f·⊙ ──► z ──► W3 行切 ──► partial y ──► all-reduce ──► y
         无通信              无通信              无通信                    ↑ 前向唯一集合通信
```

Attention 同理：QKV 列切、输出投影行切 → 子层出口 **all-reduce** 完整 hidden state。  
**每层 Transformer block 多次** all-reduce（讲义粗算系数 8×，见 [misc-llm-parallelism-table.md §5](./misc-llm-parallelism-table.md)）。

**硬依赖：** 同一 token、同一层内，all-reduce **没完成** → 下一子层 / 下一层 **不能** 用完整激活。这叫 *blocking activation communication*。

---

## 4. 「Overlap」在推理里指什么

### 4.1 三种常被混用的「重叠」

| | 含义 | 推理里 |
|--|------|--------|
| **A. 异步 collective** | `async_op=True` 发起通信；用结果前必须 `wait()` | NCCL / 运行时支持 |
| **B. 计算–通信流水线** | 传 A 的同时算 **不依赖 A 完整结果** 的别的工作 | **单请求单 token 的 decode 里很难**（见 §6） |
| **C. Ring 内部切块** | NCCL 把 all-reduce 拆成环上传块 | 库内部自动；≠ 应用层手切 $W_2$ |

通信 **语义阻塞**（没 `wait()` 不能用结果）；GPU **硬件可不傻等**（DMA 传时 SM 可干别的 **无依赖** 活）。

### 4.2 常见误区（简要）

- **「把 $W_2$ 切成 4 块，算块 2 同时发块 1」** — Megatron 应用层不这么干；那是 ring 内部或 **多请求调度**。
- **「Layer N 的 AllReduce 与 Layer N+1 前向完全并行（同一 token）」** — **不成立**；N+1 要 N 的完整 $y$。
- **推理能 overlap 的主要来源：** **连续 batching**（A 请求在算 Layer 5 时，B 请求的 collective 在飞）、**CP/EP 与计算双流**、**prefill 与 decode 批间调度**——不是单层 TP 魔法。

---

## 5. Prefill 阶段：TP / CP / EP 各发生什么

设 batch $B$ 个请求，prompt 长度 $S$（可 per-request 不同；连续 batching 下 pad 或 varlen）。

### 5.1 TP @ Prefill

- **计算：** 大 matmul，$QK^\top$ 为 $(B\!\cdot\!S,\, S)$ 量级 → **算力主导**（$S$ 大时）。
- **通信：** 每层 all-reduce **$(B, S, D)$** 激活 → **消息大**，但相对大 matmul，**通信占比常低于 decode**。
- **Overlap 机会：** matmul 时间长，**更有希望** 用 async collective 或双 stream 盖住部分 all-reduce；仍 **不能** 跳过「层间 wait 完整 hidden」。

### 5.2 CP @ Prefill

- 每卡只存 **$S/N_{\mathrm{CP}}$** 段的激活与 KV。
- Attention 需要 **看见** 别段的 K/V → 层间 **exchange / all-gather 序列分片**。
- **Prefill 是 CP 的「重通信阶段」：** $S$ 大，交换的 KV 体积 $\propto S$。

### 5.3 EP @ Prefill（MoE 层）

- Router 为 **每个 token** 选 top-$k$ 专家 → **all-to-all** 把 token 发到专家所在卡。
- Token 数 $\approx B\!\cdot\!S$ → **dispatch 体积大**；prefill 一次过，all-to-all **burst**。

```text
Prefill 墙钟 ≈ 大 matmul（TP/CP 分片后）
            + 大激活 all-reduce（TP）
            + 大 KV 交换（CP）
            + 大 token all-to-all（EP，若有 MoE）
```

---

## 6. Decode 阶段：为什么和 Prefill 完全两回事

每步只来 **1 个新 token**（先忽略 continuous batching 的 $B>1$）。

### 6.1 计算形态

```text
无 cache（教学）:  每步重算长度 1…S  的全部 Q/K/V     → O(S²) 浪费，不用
有 cache（生产）:  只算新 token 的 Q；K/V 读 cache   → O(S) 每层每步
```

- **FFN：** $(B, 1, D) \times (D, D_{\mathrm{FF}})$ — **极窄** matmul。
- **Attention：** 1 个 query 向量 attend 长度 $S$ 的 cache — **内存带宽**（读 KV）常主导。

### 6.2 TP @ Decode

- All-reduce 消息：**$(B, 1, D)$** — **字节数很小**。
- 但 **每个生成 token × 每层 × 每个 all-reduce** 都要走一次 → **延迟**（launch + 同步）堆叠。
- **结论：** Decode 下 TP 常 **「通信不重但很碎」**；算力空转 + 多卡同步 → **TP 扩展 decode 吞吐的边际收益** 往往不如 prefill。工程上靠 **continuous batching** 把 $B$ 撑大，让 all-reduce 至少不是 1 行。

**Overlap：** 单 token、单层内 **几乎无** 「大块 matmul 垫通信」窗口；overlap 主要靠 **多请求拼 batch** 或 **与 EP dispatch 流水线**。

### 6.3 CP @ Decode

- KV cache **按序列维分片** 在各 CP rank。
- 新 token 的 Q 要 attend **所有分片上的历史 K/V** → 每步仍需 **跨 rank 读 / 交换 KV**（实现依 Ring Attention、P2P gather 等）。
- **$S$ 增长 → 每步读 cache 变长**；CP 把 **每卡 KV 显存** 压住，但 **不消除** 跨卡读历史的需求。

### 6.4 EP @ Decode

- 每步只有 **少量新 token** → all-to-all **消息小**，但 **每层 MoE 仍要一次 dispatch**。
- **延迟敏感**：token 生成是串行链，MoE all-to-all 容易成 **逐步延迟** 的一部分。
- 需要足够 **$B$（并发请求）** 喂专家，否则 EP 算力空转。

```text
Decode 墙钟 ≈ Σ_{每 token} [ 读 KV（∝ S） + 窄 matmul + 多次小 all-reduce（TP）
                              + KV 跨片（CP）+ MoE all-to-all（EP）]
```

---

## 7. Prefill vs Decode：并行策略怎么选

| 痛点 | Prefill 更痛 | Decode 更痛 |
|------|--------------|-------------|
| 算力 | ✅ $S$ 大、$S^2$ attention | ❌ 窄 matmul |
| 激活/KV 显存 | ✅ 一次存整段 | ✅ cache 随 $S$ 线性涨 → **CP** |
| TP all-reduce 体积 | ✅ 大（$S$ 在形状里） | 小但 **极频繁** |
| EP all-to-all | ✅ 一次 $B\!\cdot\!S$ tokens | ❌ 每 token 一次、延迟栈 |
| 重叠空间 | matmul 大 → 相对好 | matmul 小 → **差**；靠 **batch 调度** |

**实践口诀：**

- **装不下模型权重** → **TP**（+ 可能 EP 拆专家）。
- **装不下长上下文 KV** → **CP**（prefill + decode 都受益，decode 读 cache 路径变复杂）。
- **MoE 专家太多** → **EP**（prefill 吞吐、decode 延迟都要单独 profile）。
- **Decode 慢** 时：先查 **batch 是否=1**、KV 带宽、MoE all-to-all；**不要** 默认再加 TP。

---

## 8. 小例子：2 卡 TP，FFN 出口 all-reduce

与 [misc-tp-ffn-simple.html](./misc-tp-ffn-simple.html) 一致：

```text
Prefill: S=1024, D=4096, B=1
  → all-reduce 约 2 · B · S · D 字节 ≈ 8 MB / 层 / 次（FP16）

Decode:  S=1（当前步）, 同一 D, B=1
  → all-reduce 约 8 KB / 层 / 次
  → 但 生成 512 token = 512 × L层 × 多次 all-reduce / token
```

**Prefill：** 一次 8 MB，和大 matmul 一起算 → 带宽利用率高。  
**Decode：** 一次 8 KB，kernel 启动 + NCCL 延迟占比高 → **「通信不重但很碎」**。

---

## 9. 附录：训练里才有的 TP overlap（推理可跳过）

> 推理 **没有 backward**，下列仅作对照；实现见 `cs336_systems/distributed/overlap_ddp.py`（DDP，非 TP）。

**反向** Megatron FFN 典型窗口：先 **async all-reduce($dx$)**，再算同层 **$dW$**（$dW$ 不依赖 $dx$ 已规约完）。

```text
SM:     [ dx_local ][──── dW1,dW2,dW3 ────][ wait dx ]
链路:              [ async all-reduce dx ──────]
```

Prefill 前向 **没有** 这段 $dW$ 垫后 → [tensor-parallel-calculations.md §6](../reports/tensor-parallel-calculations.md) 说 **前向更早撞通信墙**，与推理 prefill 仍偏算力、decode 偏延迟 **不矛盾**（decode 的瓶颈常在 KV 读与延迟，不在 FLOP/comm 公式本身）。

---

## 10. 复习清单

1. Prefill 与 Decode 的计算复杂度（Attention）？  
   → Prefill $\mathcal{O}(S^2)$；Decode（有 KV cache）$\mathcal{O}(S)$ **每步**。

2. 推理三种并行各切什么？  
   → TP 切层内矩阵；CP 切序列/KV；EP 切 MoE 专家。

3. TP 前向 FFN 几次 all-reduce？传什么？  
   → 出口 1 次；**激活**，非权重。

4. 为何 decode 下 TP 常「不划算」？  
   → all-reduce **很小但每层每 token 都做**；窄 matmul 盖不住同步延迟。

5. 长上下文 KV 爆显存该加谁？  
   → **CP**（不是单纯加 TP）。

6. MoE 推理主要多什么通信？  
   → 每层 **token all-to-all**；prefill burst、decode 逐步延迟。

---

## 11. 一句话总结

**推理的并行先看阶段：** Prefill 是大 $S$ 的 **算力 + 大块通信**；Decode 是小 $B\!\cdot\!1$ 的 **KV 带宽 + 碎同步**。  
**TP** 解决 **权重放不下** 和层内 matmul，但带来 **blocking all-reduce**——prefill 尚能靠大 matmul 养活，decode 则怕 **延迟栈**。  
**CP** 解决 **长序列 KV/激活显存**，prefill 与 decode 都要付 **跨片 KV 访问**。  
**EP** 解决 **MoE 专家参数**，通信是 **all-to-all**，prefill 看吞吐、decode 看逐步延迟。  
**Overlap** 在推理里主要靠 **连续 batching 与运行时调度**，不是「同一 token 上 matmul 与 AllReduce 同时出结果」。
