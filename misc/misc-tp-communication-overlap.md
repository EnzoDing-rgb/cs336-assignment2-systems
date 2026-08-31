# 推理并行：TP / CP / EP 与 Prefill–Decode 差异

> **范围：** **推理**里的 **TP / CP / EP**，以及 **Prefill vs Decode**。  
> **重点：** §6.2–§6.5 用 **Attention + FFN 一个 block** 的具体数字，算清通信、算力、以及 **continuous batching** 如何抬高 GPU 利用率。

---

## 1. 推理的两段形态：Prefill 与 Decode

同一次生成请求，GPU 上经历 **两种算力形态**：

| | **Prefill（提示词阶段）** | **Decode（逐 token 生成）** |
|--|---------------------------|------------------------------|
| **输入** | 整段 prompt，长度 $S_{\mathrm{prompt}}$（可上千） | 每步 **1 个新 token**（连续 batching 时 $B_{\mathrm{eff}}>1$） |
| **Attention** | 所有位置 **一次算完**（因果 mask） | 只算 **新 token 的 Q**；K/V 从 **KV cache 读** |
| **单层 Attention 量级** | $\mathcal{O}(S^2)$（$QK^\top$ 为 $S\times S$） | $\mathcal{O}(S)$（一个 query 对所有历史 key） |
| **典型瓶颈** | **算力**（大 matmul、Tensor Core） | **访存 / 延迟**（小 matmul + 读很长 KV） |
| **KV cache** | **写入**：为每个 prompt 位置存 K/V | **追加** 1 行 K/V；历史 K/V **只读** |
| **TP all-reduce 激活形状** | $(B,\, S,\, D)$ — **$S$ 大** | $(B,\, 1,\, D)$ — **$S$ 为 1，每层每 token 各一次** |

```text
用户 prompt ──► [ Prefill: 一口气算完 S_prompt 个 token ] ──► 采样第 1 个 token
                                                      │
                                                      ▼
                                    [ Decode: 每步 +1 token，重复 L 层 × 每 token 一步 ]
                                                      │
                                                      └── 直到 EOS / 长度上限
```

**对比记忆：**

| 阶段 | 像什么 |
|------|--------|
| Prefill | 批处理 **大矩阵乘** |
| Decode | **极窄矩阵乘** + **只读 KV 缓存**，循环成千上万次 |

**KV cache（生产推理标配）：** 每步只为 **新 token** 算 Q，并从 cache **读取** 全部历史 K/V。Decode 里 matmul 的「长度 $S$」体现在 **读 cache 的范围**，当前 Q 通常只有 **1 行**。

**对比两条路径：**

| 路径 | 每步行为 | 用途 |
|------|----------|------|
| **重算路径** | 每步重算整段 Q/K/V，复杂度 $\mathcal{O}(S^2)$ 上升 | 教学 |
| **Cache 路径** | 新 token 只算 Q，K/V 读 cache | **生产推理** |

---

## 2. 推理三种并行：各切什么

| 并行 | 切什么 | 每卡保留比例 | 主要通信 |
|------|--------|--------------|----------|
| **TP** | 层内大矩阵（列切 / 行切） | 权重 $\sim 1/N_{\mathrm{TP}}$ | 每层 block：**激活 all-reduce**（$\sim B\!\cdot\!S\!\cdot\!D$） |
| **CP** | **序列维** token / KV 分片 | 激活 + KV $\sim 1/N_{\mathrm{CP}}$ | 层间 **序列分片交换**（凑齐 attention 所需 K/V） |
| **EP** | **MoE 专家**权重 | 专家参数 $\sim 1/N_{\mathrm{EP}}$ | 每层 MoE：**token all-to-all**（dispatch + combine） |

```text
                    ┌─ TP：宽矩阵切开算
  一层 Transformer ─┼─ CP：长序列 / KV 切开存
                    └─ EP：多个专家 FFN 拆到不同卡（仅 MoE 层）

  三者可叠：节点内 TP + CP 撑长上下文 + EP 撑 MoE 参数量
```

**TP 与 CP 的分工（对比）：**

| 场景 | 更合适的选择 |
|------|--------------|
| **层内大权重** 超出单卡容量 | **TP** |
| **长序列 KV / 激活** 超出单卡容量 | **CP** |
| MoE **专家权重** 超出单卡容量 | **EP** |

**纯 TP（叠加 CP 之前）：** 每张 TP 卡持有 **整段序列** 的 hidden state；KV cache 在每张卡上 **按完整序列长度** 复制（或与 SP 组合时切中间宽激活）。**长上下文 KV 显存** 靠 **CP** 摊薄，靠加 TP 摊 **权重**。

---

## 3. TP 基础：Megatron FFN 与 All-Reduce

### 3.1 列切 + 行切（SwiGLU FFN）

单层 FFN：$x_1 = xW_1$，$x_2 = xW_2$，$z = f(x_1)\odot x_2$，$y = zW_3$。

**列并行（$W_1, W_2$）：** 按 $D_{\mathrm{FF}}$ **列**切；每张卡用 **完整** $x$ 乘本地列块 → 宽维的一块；**跨卡通信发生在 FFN 出口**。

**逐元素（GeLU / SwiGLU）：** 在本地宽维块上算，**本地完成**。

**行并行（$W_3$）：** 按 $D_{\mathrm{FF}}$ **行**切；每张卡用本地 $z$ 块乘本地行块 → 输出维的 **partial sum**，**本地完成**。

**All-Reduce（出口）：** 各卡 partial sum **求和**，每张卡得到 **完整** $y$，形状 $(B, S, D)$。这是该 FFN **前向唯一** 的集合通信。

```text
x ──► W1,W2 列切 ──► x1,x2 ──► f·⊙ ──► z ──► W3 行切 ──► partial y ──► all-reduce ──► y
         本地                 本地                 本地                    ↑ 集合通信
```

Attention 子层：**QKV 列切** → 本地 attention → **输出投影行切** → **all-reduce ①**。  
FFN 子层：**$W_1,W_2$ 列切** → 门控 → **$W_3$ 行切** → **all-reduce ②**。  
**推理前向：每层 2 次 all-reduce**（Attention 1 次 + FFN 1 次）。§6.2.1 展开 block 级数字。

### 3.2 All-Reduce 是什么

$N$ 张卡各持 partial 向量 $p^{(i)}$，all-reduce 后 **每张卡** 都得到 $y = \sum_i p^{(i)}$。

实现上常拆成两步（环形 NCCL）：

```text
all-reduce  =  reduce-scatter  +  all-gather
              （规约成 N 块）      （拼回完整张量）
```

**层间依赖：** 同一 token、同一层，all-reduce **完成后**，下一子层 / 下一层才用 **完整** 激活。这叫 **blocking activation communication**——激活通信挡在层间关键路径上。

---

## 4. Overlap 在推理里指什么

### 4.1 三种「重叠」（对比表）

| 类型 | 定义 | 推理中的情况 |
|------|------|--------------|
| **A. 异步 collective** | `async_op=True` 发起通信；用结果前调用 `wait()` | NCCL / 运行时支持 |
| **B. 计算–通信流水线** | 传数据块 A 的同时，SM 算 **独立于 A 完整结果** 的工作 | Decode 单 token 时窗口 **很窄**（§6） |
| **C. Ring 内部切块** | NCCL 把 all-reduce 拆成环上传块 | **库内部** 自动完成 |

**语义 vs 硬件（对比）：**

| 层面 | 行为 |
|------|------|
| **语义** | `wait()` 完成后，all-reduce 结果 **就绪**，下游 matmul **以此为输入** |
| **硬件** | DMA 传数据时，SM 可执行 **其他独立** 的 kernel |

餐厅类比：下单（启动通信）后可回座位（继续算别的）；**上齐菜**（`wait()`）后才进入 **以该激活为输入** 的下一层。

### 4.2 三种调度，对比谁在和谁并行

| 调度 | 并行的是 |
|------|----------|
| **Megatron 应用层** | 列/行切 **$N_{\mathrm{TP}}$ 份**；一次 all-reduce **整份** $(B,S,D)$ 激活 |
| **Ring 算法（NCCL）** | 环上 **逐块** 传同一 all-reduce 的数据 |
| **连续 batching** | 请求 A 的 Layer 5 计算 与 请求 B 的 collective **时间上交叠** |
| **同 token、相邻两层** | Layer $N+1$ 以 Layer $N$ 的 **完整** $y$ 为输入 → 两层主 matmul **串行** |

**推理 overlap 的主要来源：** 连续 batching、CP/EP 与计算双流、prefill 与 decode 批间调度——**多请求 / 多阶段** 填 GPU 空档。

---

## 5. Prefill 阶段：TP / CP / EP

设 batch $B$ 个请求，prompt 长度 $S$。

### 5.1 TP @ Prefill

| 维度 | 特征 |
|------|------|
| **计算** | 大 matmul；$QK^\top$ 为 $(B\!\cdot\!S,\, S)$ → **算力主导** |
| **通信** | 每层 all-reduce $(B, S, D)$ → **消息大**；相对大 matmul，通信占比 **小于 decode 阶段** |
| **Overlap** | matmul 耗时长 → async collective / 双 stream **有机会** 盖住部分 all-reduce；层间仍须 **wait 完整 hidden** |

### 5.2 CP @ Prefill

| 维度 | 特征 |
|------|------|
| **存储** | 每卡 **$S/N_{\mathrm{CP}}$** 段激活与 KV |
| **通信** | Attention 需全序列 K/V → 层间 **exchange / all-gather 序列分片** |
| **量级** | $S$ 大 → KV 交换体积 $\propto S$；Prefill 是 CP 的 **重通信阶段** |

### 5.3 EP @ Prefill（MoE）

| 维度 | 特征 |
|------|------|
| **路由** | 每个 token 选 top-$k$ 专家 |
| **通信** | **all-to-all** dispatch；token 数 $\approx B\!\cdot\!S$ → **burst** |

```text
Prefill 墙钟 ≈ 大 matmul（TP/CP 分片后）
            + 大激活 all-reduce（TP）
            + 大 KV 交换（CP）
            + 大 token all-to-all（EP，MoE 模型）
```

---

## 6. Decode 阶段：与 Prefill 的差异

每步 **1 个新 token**（先设 $B=1$；continuous batching 下 $B>1$ 同理放大）。

### 6.1 计算形态（对比）

| 模式 | 每步算什么 | 单层 Attention 量级 |
|------|------------|---------------------|
| **Cache 路径（生产）** | 新 token 的 Q；K/V **读 cache** | $\mathcal{O}(S)$ |
| **重算路径（教学）** | 重算长度 $1\ldots S$ 的全部 Q/K/V | $\mathcal{O}(S^2)$ 逐步上升 |

Decode 生产路径：

- **FFN：** $(B, 1, D) \times (D, D_{\mathrm{FF}})$ — **极窄** matmul
- **Attention：** 1 个 query attend 长度 $S$ 的 cache — **内存带宽**（读 KV）常主导

### 6.2 重点：一个 Block（Attention + FFN）里，通信与算力各多少？

下面用 **Megatron 风格的一个 Transformer block** 把 Prefill / Decode 算清楚。符号：

| 符号 | 含义 |
|------|------|
| $B$ | batch（同时处理的序列条数；continuous batching 下 $B>1$） |
| $S$ | Prefill：prompt 长度；Decode：**当前步新 token 数 = 1**（形状里 $S=1$） |
| $S_{\mathrm{ctx}}$ | Decode：KV cache 里 **已有上下文长度**（算力读 cache 用，**不在** all-reduce 形状里） |
| $D$ | hidden size |
| $D_{\mathrm{FF}}$ | FFN 中间维 |
| $N_{\mathrm{TP}}$ | 张量并行卡数 |
| $L$ | 模型层数 |

#### 6.2.1 Block 里 TP 切在哪、all-reduce 在哪

Megatron 规矩：**列切入口、行切出口**；每个子层 **出口 1 次** all-reduce。

```text
                    ┌── Attention ─────────────────────────────────────┐
  hidden x          │  QKV 列切 → 本地 attention（按头/列块）          │
  (B,S,D)           │  输出投影 行切 → partial                         │
                    │                              ↓                   │
                    │                    all-reduce ①  → 完整 attn 输出 │
                    └── FFN ─────────────────────────────────────────┘
                        W1,W2 列切 → 门控（本地）→ W3 行切 → partial
                                              ↓
                                    all-reduce ②  → 完整 block 输出
```

**为啥各 1 次，合计 2 次？**

| 子层 | 列切（本地 matmul） | 行切（partial sum） | all-reduce |
|------|---------------------|---------------------|------------|
| Attention | QKV 投影 | 输出投影 | **①** 出口 1 次 |
| FFN | $W_1,W_2$ 升维 | $W_3$ 降维 | **②** 出口 1 次 |

列切段 **宽维已对齐**，中间 **省掉 all-gather**；行切产生 partial，**每个子层出口 all-reduce 1 次**。FFN 的 $W_1,W_2,W_3$ **共用 1 次** 出口规约（列切 + 行切配对，见 §3）。

**讲义里的「4」与「8」（训练全步，对比用）：**

```text
推理前向 / 层：  Attention 出口 1 次 + FFN 出口 1 次  =  2 次 all-reduce
训练再 + 反向：  Attention 入口 1 次 + FFN 入口 1 次  =  再 2 次（规约 dX）
训练全步 / 层：  合计 4 次 all-reduce
环形 NCCL：      每次 all-reduce ≈ 2 段（reduce-scatter + all-gather）
讲义系数 8：     4 次 × 2 段 = 8  （乘在 bsh 上的那个 8，指训练全步，指 8 张卡）
```

**下文推理数字一律用前向 2 次 / 层：**

$$
\boxed{
\text{推理前向，每层 TP 通信（FP16 消息本体）}
= 2 \times 2 \cdot B \cdot S \cdot D
= 4\,B S D \ \text{字节}.
}
$$

每次 all-reduce 传激活 $(B,S,D)$。环形系数 $\dfrac{2(N_{\mathrm{TP}}-1)}{N_{\mathrm{TP}}}$；$N_{\mathrm{TP}}=2$ 时 $\approx 1$。

#### 6.2.2 每层算力（每 TP 卡）

**Prefill**（整段 $S$ 一次算完）：

| 子层 | 每卡 FLOPs（量级） |
|------|-------------------|
| Attention | $4 B S^2 D$（$QK^\top$ + $\mathrm{softmax}\!\cdot\! V$）+ $\dfrac{8 B S D^2}{N_{\mathrm{TP}}}$（QKV / 输出投影） |
| FFN | $\dfrac{6 B S D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$（三个 matmul） |

**Decode**（Cache 路径：新 token $S=1$，读长度 $S_{\mathrm{ctx}}$ 的 KV）：

| 子层 | 每卡 FLOPs（量级） |
|------|-------------------|
| Attention | $4 B S_{\mathrm{ctx}} D$ + $\dfrac{8 B D^2}{N_{\mathrm{TP}}}$ |
| FFN | $\dfrac{6 B D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$ |

对比 Prefill / Decode 的 **形状**：

| | Prefill | Decode |
|--|---------|--------|
| all-reduce 里的 $S$ | prompt 长度，**大** | **1** |
| matmul 的「序列维」 | $S$，**大** | **1**（FFN）；Attention **读** $S_{\mathrm{ctx}}$ cache |

---

### 6.3 算例：Llama 量级，Prefill vs Decode（$B=1$）

取 $D=4096$，$D_{\mathrm{FF}}=14336$，$N_{\mathrm{TP}}=2$，$L=32$，FP16。

#### Prefill：$B=1$，$S=2048$

**一层通信：**

$$
4 \cdot 1 \cdot 2048 \cdot 4096 = 33.5\ \text{MB} \approx 34\ \text{MB}.
$$

**整模 Prefill 通信（32 层）：**

$$
32 \times 34\ \text{MB} \approx 1.1\ \text{GB}\ \text{（一次 prefill pass）}.
$$

**一层算力（每卡，加总 Attn + FFN）：**

```text
Attention:  4·B·S²·D  +  8·B·S·D²/N_TP
          ≈ 4·2048²·4096  +  8·2048·4096²/2
          ≈  69 GFLOP  +  137 GFLOP  ≈  206 GFLOP

FFN:      6·B·S·D·D_FF/N_TP
          ≈ 6·2048·4096·14336/2
          ≈  361 GFLOP

合计 ≈ 567 GFLOP / 层
```

**整模 Prefill 算力：** $567 \times 32 \approx 18$ TFLOP / 卡。

**墙钟直觉（设有效算力 150 TFLOP/s，链路 100 GB/s）：**

| 项 | 估算 |
|----|------|
| 计算 | $18\ \text{TFLOP} / 150\ \text{TFLOP/s} \approx 120\ \text{ms}$ |
| 通信（若串行暴露） | $1.1\ \text{GB} / 100\ \text{GB/s} \approx 11\ \text{ms}$ |

Prefill：**大 matmul 与大 all-reduce 同 pass**；算力与带宽 **同一量级**，Tensor Core 容易 **吃满**。

#### Decode：$B=1$，$S=1$，$S_{\mathrm{ctx}}=2048$（每步 1 token）

**一层通信（2 次 all-reduce，合计）：**

$$
4 \cdot 1 \cdot 1 \cdot 4096 = 16\ \text{KB}.
$$

**每步生成 1 token（32 层 × 2 次/层）：**

$$
32 \times 16\ \text{KB} = 512\ \text{KB / step};\qquad \text{collective 次数} = 2 \times 32 = 64\ \text{次/step}.
$$

**一层算力（每卡）：**

```text
Attention:  4·B·S_ctx·D  +  8·B·D²/N_TP
          ≈ 4·2048·4096  +  8·4096²/2
          ≈  34 MFLOP  +  67 MFLOP  ≈  101 MFLOP

FFN:      6·B·D·D_FF/N_TP
          ≈ 6·4096·14336/2
          ≈  176 MFLOP

合计 ≈ 277 MFLOP / 层  →  整步 ≈ 8.9 GFLOP / 卡
```

**墙钟直觉——Decode 的两条腿：**

| 腿 | $B=1$ 时发生什么 |
|----|------------------|
| **带宽** | 512 KB/step ÷ 100 GB/s ≈ **0.005 ms**（纯传字节极快） |
| **延迟** | **64 次** collective / step（每层 Attention 1 次 + FFN 1 次）；每次 launch + 同步常 **10–50 μs** → 光延迟 **0.6–3 ms**，与整步 **≈9 GFLOP** 的窄 matmul **同阶** |

Decode 的 TP 画像：**单次消息 16 KB（每层合计）**，**64 次/step**；瓶颈在 **延迟栈 + 窄 matmul**，带宽仍富余。

---

### 6.4 算例：continuous batching 把 $B$ 从 1 拉到 32

同样 Decode：$S=1$，$S_{\mathrm{ctx}}=2048$，$L=32$，$N_{\mathrm{TP}}=2$。

#### 通信：随 $B$ **线性放大**

| $B$ | 单层通信 $4BSD$ | 每 step 全模型 | collective 次数 |
|-----|-----------------|----------------|-----------------|
| 1 | 16 KB | 512 KB | 64 |
| 32 | **512 KB** | **16 MB** | 64（相同） |

collective **次数** 仍是 **64 / step**；**每次消息** 随 $B$ 线性变宽 → 链路从 **延迟主导** 移向 **带宽主导**。

#### 算力：也随 $B$ **线性放大**

| $B$ | 每 step 每卡 FLOPs（≈ 277 MFLOP/层 × 32 层 × $B$） |
|-----|-----------------------------------------------------|
| 1 | ≈ 8.9 GFLOP |
| 32 | ≈ **285 GFLOP** |

FFN matmul 形状从 $(1, D) \times (D, D_{\mathrm{FF}})$ 变为 $(32, D) \times (D, D_{\mathrm{FF}})$——**行数 = batch**，Tensor Core **更容易打满**。

#### 为什么加大 $B$ 让算力吃得更满？（三句话）

```text
① 固定开销摊薄：64 次 collective / step 的次数不变；B 变大 → 每次 all-reduce 多传 B 行激活。

② matmul 变「胖」：Decode FFN 的 m 维从 1 → B；Tensor Core 吃 $(B,D)$ 比 $(1,D)$ 满。

③ 吞吐换延迟：continuous batching 下一次 step 服务 B 个请求，tokens/s 随 B 近线性涨。
```

**对比表（Decode，$S_{\mathrm{ctx}}=2048$）：**

| | $B=1$ | $B=32$ |
|--|-------|--------|
| 单层通信（2 次 AR 合计） | 16 KB | 512 KB |
| 每 step collective 次数 | **64** | **64** |
| 每 step 算力 | ≈ 9 GFLOP | ≈ 285 GFLOP |
| 典型瓶颈 | **延迟 + 窄 matmul** | **带宽 + 算力** |
| continuous batching | 基准 | matmul 与 all-reduce **同 Prefill 一样变胖** |

---

### 6.5 Prefill vs Decode：TP 对照（汇总）

| 维度 | Prefill | Decode（$B=1$） | Decode（$B=32$，continuous batching） |
|------|---------|-----------------|----------------------------------------|
| all-reduce 形状 | $(B, S, D)$，$S$ 大 | $(B, 1, D)$ | $(32, 1, D)$ |
| 单层通信 $4BSD$ | 34 MB（$S{=}2048$） | 16 KB | 512 KB |
| all-reduce 次数 / token | $2L$（每层 2 次） | $2L=64$ | 同左，消息 ×32 |
| 单层算力 | ≈ 567 GFLOP/层 | ≈ 277 MFLOP/层 | ≈ 8.9 GFLOP/层（整步 ≈ 285 GFLOP） |
| 瓶颈 | 带宽 + 算力 | **延迟栈** | 带宽 + 算力（改善） |
| 加 TP 的边际收益 | **高**（权重分摊 + 大 matmul） | **低**（消息轻、次数多） | **回升**（$B$ 撑宽消息与 matmul） |

**Overlap（Decode）：** $B=1$ 时单层 matmul **块小**，垫通信窗口 **窄**；continuous batching 把 **多请求拼进同一个 $B$**，与 **EP dispatch 流水线** 一起填 GPU 空档。

### 6.6 CP @ Decode

- KV cache **按序列维分片** 在各 CP rank
- 新 token 的 Q attend **各分片** 上的历史 K/V → 每步 **跨 rank 读 / 交换 KV**
- $S$ 增长 → 每步读 cache **变长**；CP **压低每卡 KV 显存**，跨卡读历史 **仍是 decode 路径的一部分**

### 6.7 EP @ Decode

- 每步 **少量新 token** → all-to-all **消息小**，每层 MoE **仍 dispatch 一次**
- Token 生成 **串行链** → MoE all-to-all 计入 **逐步延迟**
- 需足够 **并发 $B$** 喂专家，负载才满

```text
Decode 墙钟 ≈ Σ_{每 token} [ 读 KV（∝ S） + 窄 matmul + 多次小 all-reduce（TP）
                              + KV 跨片（CP）+ MoE all-to-all（EP）]
```

---

## 7. Prefill vs Decode：策略对照

| 维度 | Prefill 侧 | Decode 侧 |
|------|------------|-----------|
| 算力 | $S$ 大，Attention $\mathcal{O}(S^2)$ **主导** | FFN 窄 matmul；Attention **读** $S_{\mathrm{ctx}}$ cache |
| 激活 / KV 显存 | 一次存整段 prompt | cache 随 $S_{\mathrm{ctx}}$ 线性涨 → **CP** |
| TP all-reduce | $4BSD$，$S$ 大 | $4BD$（$S{=}1$）；**$2L$ 次/token** |
| EP all-to-all | 一次 $B\!\cdot\!S$ tokens **burst** | 每 token 一次，逐步累加 |
| continuous batching | 天然大 $B$、大 $S$ | **抬高 $B$** → 消息与 matmul 同 Prefill 一样变「胖」 |

| 目标 | 手段 |
|------|------|
| **层内大权重** 超出单卡容量 | **TP**（MoE 叠加 **EP**） |
| **长上下文 KV** 超出单卡容量 | **CP** |
| MoE **专家参数量** 超出单卡容量 | **EP** |
| Decode 吞吐偏低 | **continuous batching 增大 $B$**；查 KV 带宽与 MoE all-to-all |

---

## 8. 推理 vs 训练：TP 能不能 overlap？

**结论先放前面：**

| 路径 | 每层 all-reduce 次数 | 层内 overlap 窗口 | 主要 overlap 来源 |
|------|----------------------|-------------------|-------------------|
| **推理 Prefill 前向** | **2**（Attn + FFN 出口） | 窄：大 matmul 与 AR 仍 **串行依赖** | 多请求 batch、双流 async |
| **推理 Decode 前向** | **2L / token**（64 次/step，$L{=}32$） | 极窄：matmul 块小 | **continuous batching** 撑大 $B$ |
| **训练 前向** | **2**（同推理） | 同推理 Prefill | 同左 |
| **训练 反向** | **再 +2**（Attn + FFN **入口**规约 $dx$） | **宽：$dx$ async + 同层 $dW$ 并行** | Megatron 调度 |

---

### 8.1 推理：同一 token、同一层上，matmul 与 all-reduce **串行**

行切 matmul 产出 partial → **必须 all-reduce** → 下一子层才用 **完整** 激活。时间线（单层 Attention 或 FFN 出口）：

```text
SM:     [── 行切 matmul（partial）──][ wait ][ 下一子层 matmul ]
链路:                              [ all-reduce ]
```

**async collective 能做什么：** 发起 all-reduce 后 SM 可去算 **别的请求 / 别的层 / 别的 stream 上无依赖的 kernel**；**本 token 本层** 的下一 matmul 仍要在 `wait()` 之后。

---

### 8.2 推理 Prefill：overlap 空间

| 因素 | 说明 |
|------|------|
| **消息大小** | 每次 AR 传 $(B,S,D)$，$S$ 大 → 单次 **16 MB 量级/层**（2 次合计 34 MB） |
| **算力** | 单层 **≈567 GFLOP** → matmul 耗时长 |
| **overlap** | 大 matmul 期间 **async 发起** AR，或 **compute stream / comm stream** 并行；**层间 wait** 仍在 |
| **对比 Decode** | Prefill 的 AR **/message** 大，带宽利用率高；Decode 的 AR **/message** 小，**次数** 同为 2/层/token |

Prefill 墙钟近似：

$$
T_{\mathrm{prefill}} \approx \sum_{\mathrm{layers}}\bigl(T_{\mathrm{matmul}} + T_{\mathrm{AR}}\bigr)\ \text{（层间串行）}
$$

大 $T_{\mathrm{matmul}}$ 让 $T_{\mathrm{AR}}$ 相对 **易被盖住一部分**（双 stream 下取 $\max$ 的项变少）。

---

### 8.3 推理 Decode：overlap 空间

| 因素 | $B=1$ | $B=32$（continuous batching） |
|------|-------|--------------------------------|
| 每层 AR 合计 | 16 KB | 512 KB |
| 每 step 次数 | **64** | **64** |
| 每 step 算力 | ≈ 9 GFLOP | ≈ 285 GFLOP |
| 瓶颈 | **64 次 launch 延迟** | 带宽 + 算力（改善） |

Decode 的 TP overlap **主要靠抬高 $B$**：次数 **64 不变**，单次消息与 matmul **同乘 $B$**——与 Prefill 同构的「胖消息 + 胖 matmul」。

**推理侧 overlap 三板斧（对比训练）：**

```text
① continuous batching     → 同一 step 多请求，B>1
② 调度器填缝              → prefill batch 与 decode batch 交替占 SM
③ comm / compute 双流     → async AR 与无依赖 kernel 并行
```

训练独有的 **$dx$ async + $dW$ 并行**（§8.4），推理 **用不上**——推理路径 **只有前向**。

---

### 8.4 训练：反向多出的 overlap 窗口

训练全步每层 **4 次** all-reduce：前向 2 次（Attn/FFN 出口）+ 反向 2 次（Attn/FFN 入口，规约 $dx$）。

**FFN 反向（Megatron）：**

```text
① dW3, dz, dx2, dx1        ← 本地
② dx_local = dx1·W1ᵀ + dx2·W2ᵀ
③ all-reduce(dx_local) → dx   ← 要传给更前面的层
④ dW1, dW2                  ← 只依赖 x, dx1, dx2；与 ③ 结果无关
```

| 调度 | 时间线 |
|------|--------|
| **串行** | ① → ② → ③ wait → ④ |
| **重叠（训练常用）** | ① → ② → **async ③** → ④（与 ③ 并行）→ wait ③ → 上一层 |

```text
SM:     [──①②──][──── ④ dW1,dW2（大 matmul）────][ wait ][ 上一层 bwd ]
链路:            [──── async all-reduce dx ────────]
                      ↑ ④ 与 ③ 并行：推理前向没有 ④ 这一步
```

Attention 反向 **对称**：输出投影入口规约 $dx$，与 QKV 的 $dW$ 可同样并行。

**训练 vs 推理 overlap 对照：**

| | 推理 Prefill | 推理 Decode | 训练（前+反） |
|--|--------------|-------------|---------------|
| 层内「算 $dW$ 垫 $dx$ AR」 | 无（仅前向） | 无 | **有** |
| 前向 AR 与下一 matmul | 串行 wait | 串行 wait | 串行 wait |
| 抬高吞吐的主招 | 大 $S$ 天然胖 | **continuous batching** | 大 batch + backward overlap |
| 每层 AR 次数 / 全步 | 2（前向） | 2（前向）× 每 token | **4**（前 2 + 反 2） |

---

### 8.5 一张图记推理 TP overlap

```text
                    推理能 overlap 的          推理层内仍串行的
                    ─────────────────          ─────────────────
Prefill:            双流 / 多请求              partial → AR → 下一子层
Decode B=1:         几乎只有调度填缝           64 次 AR/step，消息 16KB/层
Decode B=32:        消息 512KB/层，matmul 胖   同上，但带宽利用率↑
训练反向:           dx async + dW 并行         （推理路径不含此支）
```

---

## 9. 复习清单

1. 推理前向，一个 block 几次 TP all-reduce？每层合计消息？  
   → **2 次**（Attention 出口 1 + FFN 出口 1）；每层 **$4BSD$** 字节（FP16）。

2. Prefill $B{=}1,S{=}2048,D{=}4096$：单层？32 层 pass？  
   → **≈34 MB/层**；Prefill pass **≈ 1.1 GB**。

3. Decode $B{=}1$：单层？每 step？collective 次数？  
   → **16 KB/层**；**512 KB/step**；**64 次/step**（$2L$，$L{=}32$）。

4. 讲义「4 次 / 8 系数」指什么？  
   → **训练全步** 每层 4 次 AR（前 2 + 反 2）；**8 = 4 × 环形 2 段**。

5. continuous batching $B{=}1\to32$？  
   → 次数 **64 不变**；单层消息 **16 KB → 512 KB**；matmul 行维 ×32。

6. 推理 vs 训练，TP overlap 差在哪？  
   → 训练反向 **$dx$ async + $dW$ 并行**；推理靠 **continuous batching + 双流**。

7. TP / CP / EP 各切什么？  
   → TP：层内矩阵；CP：序列/KV；EP：MoE 专家。

---

## 10. 总结

| 阶段 | 主导矛盾 | TP | CP | EP |
|------|----------|----|----|-----|
| **Prefill** | 算力 + 大块通信 | 大 all-reduce，大 matmul **同 pass** | 大 KV 交换 | 大 dispatch burst |
| **Decode** | KV 带宽 + 延迟栈 | **2L 次/token**；$B{=}1$ 时 **512 KB/step** | 跨片读 cache | 逐步 all-to-all |

**Decode 核心数字（$D{=}4096,S_{\mathrm{ctx}}{=}2048,L{=}32$）：** $B=1$ → **512 KB/step、64 次 AR、≈9 GFLOP**；$B=32$ → **16 MB/step、64 次 AR、≈285 GFLOP**。

**Overlap：** 推理靠 **continuous batching** 撑宽 AR 与 matmul；训练 **额外** 靠反向 **$dx$ async + $dW$**（§8）。层内 partial → AR → 下一子层 **始终串行**。
