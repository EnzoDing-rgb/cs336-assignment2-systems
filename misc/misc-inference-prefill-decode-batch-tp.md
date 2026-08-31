# 推理：Prefill、Decode 与 Batch Size（先无 TP，再有 TP）

> **读法：** 先假设 **整模型在单卡上**（$N_{\mathrm{TP}}=1$，无跨卡 all-reduce），把 Prefill / Decode / Batch 的关系算清楚；再叠 **Tensor 并行**，看通信税与量化阈值。  
> **模型常量（全文共用）：** $D=4096$，$D_{\mathrm{FF}}=14336$，$L=32$；有效算力 $C=150\ \mathrm{TFLOP/s}$（Prefill 峰值附近），Decode 小矩阵有效 $C_{\mathrm{eff}}\approx 30\ \mathrm{TFLOP/s}$。

---

## 0. 符号

| 符号 | 含义 |
|------|------|
| $B$ | 同时服务的序列条数（continuous batching） |
| $S$ | Prefill：prompt 长度 |
| $S_{\mathrm{ctx}}$ | Decode：KV cache 已有长度 |
| $N_{\mathrm{TP}}$ | 张量并行度；**第一节固定为 1** |
| $C,\,C_{\mathrm{eff}}$ | 峰值 / 小矩阵有效算力 |
| $W$ | 单卡出向带宽（**仅第二节 TP 用**） |
| $\tau$ | 单次 kernel / collective 固定延迟（**10–50 μs**） |

---

# 第一节 · 无 TP（$N_{\mathrm{TP}}=1$）

单卡持有 **完整权重**；前向 **零跨卡通信**。瓶颈只有：**算力、matmul 形状、KV 读带宽、每层 kernel 启动**。

---

## 1.1 Prefill vs Decode：算力公式

**Prefill**（一次算完 prompt，每层每卡）：

$$
F_{\mathrm{prefill,layer}}
=
4 B S^2 D
+
8 B S D^2
+
6 B S D D_{\mathrm{FF}}.
$$

**Decode**（KV cache，每层每卡，每 step）：

$$
F_{\mathrm{decode,layer}}
=
4 B S_{\mathrm{ctx}} D
+
8 B D^2
+
6 B D D_{\mathrm{FF}}.
$$

**matmul 的「行维」$m$：**

| 阶段 | 主 matmul 行维 | 谁提供「胖矩阵」 |
|------|----------------|------------------|
| Prefill | $m = B \cdot S$ | **$S$**（prompt 长度） |
| Decode | $m = B$ | **只有 $B$** |

---

## 1.2 算例：$B=1$ 已经够胖 vs 必须靠 Batch

### Prefill：$B=1$，$S=2048$

```text
Attention:  4·2048²·4096  +  8·2048·4096²   ≈  275 GFLOP
FFN:        6·2048·4096·14336              ≈  722 GFLOP
合计 ≈ 997 GFLOP/层  →  整 pass ≈ 32 TFLOP

m = B·S = 2048
T_compute ≈ 32 TFLOP / 150 TFLOP/s ≈ 210 ms
```

**$B=1$ 时 $m=2048$**，Tensor Core 已处于高效区；**Prefill 不需要为了「把矩阵乘胖」去排大 Batch**。

### Decode：$B=1$，$S_{\mathrm{ctx}}=2048$

```text
Attention:  4·2048·4096  +  8·4096²        ≈  168 MFLOP
FFN:        6·4096·14336                    ≈  353 MFLOP
合计 ≈ 521 MFLOP/层  →  整 step ≈ 16.7 GFLOP

m = 1
T_peak ≈ 16.7 GFLOP / 150 TFLOP/s ≈ 0.11 ms   ← 峰值，达不到
T_eff  ≈ 16.7 GFLOP /  30 TFLOP/s ≈ 0.56 ms   ← 小矩阵有效算力
```

**$B=1$ 时 $m=1$**，matmul 极窄；算力仅为 Prefill 同 $B$ 下单层的 **~1/2000**。

### Decode：$B=32$

```text
整 step FLOPs ≈ 16.7 × 32 ≈ 534 GFLOP
m = 32
T_eff ≈ 534 GFLOP / 30 TFLOP/s ≈ 18 ms
```

**加大 $B$ 把 $m$ 从 1 拉到 32**，Decode 才进入「矩阵够宽、核够满」的区间。

---

## 1.3 无 TP 时 Decode 为何要 continuous batching

除 matmul 行维外，单卡 Decode 还有 **固定开销**（每层一次前向链，$L$ 层 / step）：

$$
T_{\mathrm{step,fix}} \approx L \cdot \tau_{\mathrm{kern}}
\quad\text{（$\tau_{\mathrm{kern}}$：每层调度 + kernel 启动，经验 10–30 μs）}.
$$

$L=32$，$\tau_{\mathrm{kern}}=20\ \mu s$ → $T_{\mathrm{fix}} \approx 0.64\ \mathrm{ms}$。

| | $B=1$ | $B=32$ |
|--|-------|--------|
| 有效算力时间 $T_{\mathrm{eff}}$ | ≈ 0.56 ms | ≈ 18 ms |
| 固定开销 $T_{\mathrm{fix}}$ | ≈ 0.64 ms | ≈ 0.64 ms |
| 合计粗估 | **~1.2 ms/step** | **~19 ms/step** |
| 每 token 摊销（32 序列并行） | 1.2 ms / 1 | 19 ms / 32 ≈ 0.6 ms/token 吞吐 |

**无 TP 临界 batch（算力 + 固定开销）：**

$$
\boxed{
B_{\mathrm{crit}}^{\mathrm{(no TP)}}
\;\approx\;
\frac{L\,\tau_{\mathrm{kern}}}{F_{\mathrm{step}}(B{=}1)/C_{\mathrm{eff}}}
\;\approx\;
\frac{L\,\tau_{\mathrm{kern}}}{T_{\mathrm{eff}}(B{=}1)}.
}
$$

代入：$T_{\mathrm{eff}}(1)\approx 0.56\ \mathrm{ms}$，$L\tau_{\mathrm{kern}}\approx 0.64\ \mathrm{ms}$ → $B_{\mathrm{crit}}^{\mathrm{(no TP)}} \approx 1\text{–}8$（偏保守）；工程上仍常取 **$B \gtrsim 16\text{–}32$**，因还要 **KV 读带宽、多请求吞吐** 一并考虑。

**matmul 效率阈值（另一条线）：** Tensor Core 高效区常需 $m \gtrsim 64$ → **Decode 目标 $B \gtrsim 64$** 才与 Prefill 的 $m=2048$ 同量级效率（Prefill 用 $S$，Decode 用 $B$）。

---

## 1.4 无 TP 对比总表

| | **Prefill** | **Decode** |
|--|-------------|------------|
| matmul 行维 | $B \cdot S$ | $B$ |
| $B=1$ 典型 $m$ | 2048（$S=2048$） | **1** |
| $B=1$ 整 pass/step 算力 | **≈32 TFLOP** | **≈17 GFLOP** |
| 靠谁变胖 | **$S$** | **$B$** |
| 是否需要大 Batch | **单用户 $S$ 已够** | **需要 continuous batching** |
| 跨卡通信 | **无** | **无** |

**第一节结论（无 TP）：** Prefill 的 batch 维在 **$S$**；Decode 的 batch 维只能靠 **$B$**。与是否开 TP **无关**——这是 **matmul 形状** 的结构性差异。

---

## 1.5 无 TP 时要不要「并行」？

单卡能装下 **权重 + KV + 激活** → **$N_{\mathrm{TP}}=1$ 即可**，Prefill 与 Decode 逻辑 **同上表**。

**权重粗算（FP16）：**

$$
M_{\mathrm{weights}} \approx L(4D^2 + 3DD_{\mathrm{FF}})\cdot 2
\approx 16\ \mathrm{GB}\quad (D{=}4096,\,D_{\mathrm{FF}}{=}14336,\,L{=}32).
$$

| 显存 | 无 TP |
|------|-------|
| 80 GB 卡，16 GB 权重 | **$N_{\mathrm{TP}}=1$**；第二节 TP 为 **可选项** |
| 70B（$\sim$140 GB 权重） | 单卡装不下 → **必须进入第二节 TP** |

**无 TP 时不存在 all-reduce 阈值**；Decode 仍按 **$B_{\mathrm{crit}}^{\mathrm{(no TP)}}$** 排 batch。

---

# 第二节 · 有 TP（$N_{\mathrm{TP}} \ge 2$）

在第一节 **同一套算力公式** 上，权重按 Megatron **切分到 $N_{\mathrm{TP}}$ 卡**；每层前向 **+2 次 all-reduce**（Attention 出口 + FFN 出口）。

---

## 2.1 相对无 TP，多了什么

| 项 | 无 TP | 有 TP |
|----|-------|-------|
| 权重 / 卡 | 全量 $M_{\mathrm{weights}}$ | $\approx M_{\mathrm{weights}}/N_{\mathrm{TP}}$ |
| 层内 FFN / 投影 FLOPs | 全量 | $\approx \times 1/N_{\mathrm{TP}}$（宽维切分） |
| Attention $BS^2D$ 项 | 全量 | **各卡仍算完整**（按头/块切，量级同阶） |
| 跨卡通信 | 0 | 每层 **$4BSD$** 字节（FP16，2 次 AR） |
| Decode 每 step 通信次数 | 0 | **$2L$** 次 |

**有 TP 的每层算力（每卡）：**

$$
F_{\mathrm{prefill,layer}}^{\mathrm{(TP)}}
\approx
4 B S^2 D
+
\frac{8 B S D^2 + 6 B S D D_{\mathrm{FF}}}{N_{\mathrm{TP}}},
\qquad
F_{\mathrm{decode,layer}}^{\mathrm{(TP)}}
\approx
4 B S_{\mathrm{ctx}} D
+
\frac{8 B D^2 + 6 B D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
$$

**Attention 的 $4BS^2D$（Prefill）与 $4BS_{\mathrm{ctx}}D$（Decode）不除 $N_{\mathrm{TP}}$** → TP **主要省 FFN/投影**，**不省** Attention 二次项（Prefill）或 cache 读取（Decode）。

---

## 2.2 有 TP 的通信公式

每层 2 次环形 all-reduce，消息各 $2BSD$ 字节：

$$
T_{\mathrm{comm,layer}}^{\mathrm{(TP)}}
=
\frac{8\,(N_{\mathrm{TP}}-1)\,B S D}{N_{\mathrm{TP}}\,W}.
$$

Decode（$S=1$）整 step：

$$
M_{\mathrm{step}}^{\mathrm{(TP)}} = 4 B D L\ \text{字节},\qquad
N_{\mathrm{call}} = 2L,\qquad
T_{\mathrm{comm,step}}^{\mathrm{(TP)}}
=
\frac{8\,(N_{\mathrm{TP}}-1)\,B D L}{N_{\mathrm{TP}}\,W}.
$$

**相对无 TP 新增延迟：** $N_{\mathrm{call}}\cdot\tau$（NCCL collective，$\tau \approx 20\ \mu s$）。

---

## 2.3 算例：$N_{\mathrm{TP}}=2$ 与无 TP 对照

### Prefill：$B=1$，$S=2048$

| | 无 TP | 有 TP（$N_{\mathrm{TP}}=2$） |
|--|-------|------------------------------|
| 算力 / pass | ≈ 32 TFLOP | ≈ 18 TFLOP（FFN 减半，Attn 仍大） |
| 通信 | 0 | ≈ 1.1 GB / pass |
| $T_{\mathrm{compute}}$ | ≈ 210 ms | ≈ 120 ms |
| $T_{\mathrm{comm}}$ | 0 | ≈ 11 ms（$W=100$ GB/s） |

**Prefill + TP：** $S$ 仍提供 $m=2048$；通信相对算力 **小** → **$B=1$ 仍够**。

### Decode：$B=1$，$S_{\mathrm{ctx}}=2048$

| | 无 TP | 有 TP（$N_{\mathrm{TP}}=2$） |
|--|-------|------------------------------|
| 算力 / step | ≈ 17 GFLOP | ≈ 8.9 GFLOP |
| 通信 / step | 0 | **512 KB** |
| collective / step | 0 | **64 次** |
| 新增 $N_{\mathrm{call}}\tau$ | — | **≈ 1.3 ms** |

**Decode + TP：** 在无 TP 已偏窄的 matmul 上，再叠 **64 次 collective** → **更依赖大 $B$**。

### Decode：$B=32$（有 TP）

| | 无 TP | 有 TP |
|--|-------|-------|
| 算力 / step | ≈ 534 GFLOP | ≈ 285 GFLOP |
| 通信 / step | 0 | **16 MB** |
| collective 次数 | — | **64**（不变） |

**$B$ 翻倍消息与算力；collective 次数不变** → 摊薄 **单次 AR 延迟**。

---

## 2.4 有 TP：Prefill / Decode 量化判定

### （A）硬条件：必开 TP

$$
\boxed{
N_{\mathrm{TP}} \;\ge\;
\left\lceil \frac{M_{\mathrm{weights}}}{M_{\mathrm{GPU}}} \right\rceil.
}
$$

Prefill 与 Decode **共用权重** → **同一 $N_{\mathrm{TP}}$**。

### （B）Prefill：通信是否压算力（$B$ 约掉）

令 $T_{\mathrm{comm,layer}}^{\mathrm{(TP)}} \ge T_{\mathrm{comp,layer}}^{\mathrm{(TP)}}$，Attention 主项 $4BS^2D$ 主导：

$$
\boxed{
S \;\le\;
S_{\mathrm{crit,prefill}}^{\mathrm{(TP)}}
=
\sqrt{\frac{2\,(N_{\mathrm{TP}}-1)\,C}{N_{\mathrm{TP}}\,W\,D}}.
}
$$

$N_{\mathrm{TP}}=2$，$C=150$ TFLOP/s，$W=100$ GB/s → **$S_{\mathrm{crit,prefill}}^{\mathrm{(TP)}} \approx 192$**。

| $S$ | 含义 |
|-----|------|
| $\gtrsim 192$ | Prefill **算力主导**；**$B=1$ 即可**（与无 TP 结论一致，多付通信） |
| $\ll 192$ | TP 通信 **相对重**；仍可能因 **装权重** 必须 TP |

### （C）Decode：通信 vs 读 cache 算力（$B$ 约掉）

$$
\boxed{
S_{\mathrm{ctx}} \;\le\;
S_{\mathrm{crit,decode}}^{\mathrm{(TP)}}
=
\frac{2\,(N_{\mathrm{TP}}-1)\,C}{N_{\mathrm{TP}}\,W}
\;\approx\; 1500\ \text{tokens}\ \text{（上例）}.
}
$$

| $S_{\mathrm{ctx}}$ | 含义 |
|--------------------|------|
| $\gtrsim 1500$ | 读 KV 算力 **渐增**，养得起 TP 通信 |
| $\ll 1500$ | TP 通信 **相对重** |

### （D）Decode：Batch 阈值（叠在无 TP 上）

$$
\boxed{
B \;\ge\;
B_{\mathrm{crit,decode}}^{\mathrm{(TP)}}
\;\approx\;
\max\!\left(
B_{\mathrm{crit}}^{\mathrm{(no TP)}},\;
\frac{N_{\mathrm{call}}\,\tau}{T_{\mathrm{eff}}^{\mathrm{(TP)}}(B{=}1)}
\right).
}
$$

| 来源 | 量级 |
|------|------|
| 无 TP：matmul 要胖 | **$B \gtrsim 16\text{–}64$** |
| 有 TP：摊 **$2L$ 次** collective | **$B \gtrsim 16\text{–}32$**（$N_{\mathrm{call}}\tau \approx 1.3\ \mathrm{ms}$） |

**Decode 排 batch：先满足第一节（$m$ 维），有 TP 时再抬 $B$ 摊 collective。**

### （E）$N_{\mathrm{TP}}$ 开多大（Prefill 能否线性加速）

$$
\boxed{
N_{\mathrm{TP}} \;\ge\;
1 + \frac{3}{2}\,D_{\mathrm{FF}}\,\frac{W}{C}
\;\approx\; 15
\quad\Rightarrow\quad
\text{再加 TP 卡，Prefill 通信开始压算力}.
}
```

这是 **「加卡是否还加速」**，与 **「要不要 TP 装模型」** 分开。

---

## 2.5 有 TP 决策流程

```text
【无 TP 基线】单卡能装权重？
        │
       是 ──→ 第一节：Prefill 靠 S，Decode 靠 B（可 N_TP=1）
        │
       否 ──→ 进入 TP，N_TP ≥ ⌈M_w / M_GPU⌉
                    │
        ┌───────────┴───────────┐
   Prefill                    Decode
   S ≳ S_crit_prefill ?       B ≳ B_crit_decode ?
   （~192）                    （~16–32，叠 collective）
        │                           │
   第一节结论仍成立：            第一节 + 摊 2L 次 AR
   S 养算力；B=1 够            B 养 matmul + 养 TP 通信
        └───────────┬───────────┘
                    │
           两阶段同一 N_TP 组
```

---

## 2.6 速查：无 TP vs 有 TP

| 问题 | **无 TP** | **有 TP** |
|------|-----------|-----------|
| Prefill 要大 $B$ 吗？ | **$S$ 够**（$m=B\!\cdot\!S$） | **同左**；多 **$4BSDL$** 通信 |
| Decode 要大 $B$ 吗？ | **要**（$m=B$） | **更要**（+ **$2L$ 次 AR**） |
| 何时必须用 TP？ | 单卡装不下权重 | **$N_{\mathrm{TP}} \ge \lceil M_w/M_{\mathrm{GPU}}\rceil$** |
| Prefill 通信阈值 | — | $S \gtrsim \sqrt{2(N{-}1)C/(NWD)}$ |
| Decode 通信阈值 | — | $S_{\mathrm{ctx}} \gtrsim 2(N{-}1)C/(NW)$ |
| Decode batch 阈值 | $B \gtrsim 16\text{–}64$ | $B \gtrsim 16\text{–}32$（含摊 AR） |

---

## 复习清单

1. **无 TP**：Prefill $m=$？Decode $m=$？ → $B\!\cdot\!S$ vs $B$。  
2. **无 TP**：$B=1,S=2048$ Prefill vs Decode 算力？ → **≈32 TFLOP** vs **≈17 GFLOP/step**。  
3. **有 TP**：每层几次 AR？Decode 每 step 几次？ → **2 / 层**；**$2L$**（如 64）。  
4. **有 TP**：Prefill 仍不需大 $B$ 的原因？ → **$S$ 养算力**（与无 TP 相同）。  
5. **有 TP**：Decode 更要 batch 的原因？ → 第一节 $m=B$ **+** 摊 **$2L\tau$**。
