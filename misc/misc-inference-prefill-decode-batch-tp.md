# 推理：Prefill、Decode 与 Batch Size —— 算力强度与 Tensor 并行判定

> **范围：** 推理前向；并行只看 **TP**（张量并行）。  
> **两个问题：** ① Prefill / Decode 的 **计算强度** 为何不同 → Batch Size 需求不同；② 何时 **值得开 TP** → 给出 **$B$、$S$、$N_{\mathrm{TP}}$** 的量化阈值。

---

## 0. 符号

| 符号 | 含义 |
|------|------|
| $B$ | 同时服务的序列条数（continuous batching 的 batch） |
| $S$ | Prefill：prompt 长度；Decode：all-reduce 形状里的序列维 **= 1** |
| $S_{\mathrm{ctx}}$ | Decode：KV cache 已有长度（算力读 cache 用） |
| $D$ | hidden size |
| $D_{\mathrm{FF}}$ | FFN 中间维 |
| $L$ | Transformer 层数 |
| $N_{\mathrm{TP}}$ | 张量并行度 |
| $C$ | 单卡有效算力（FLOP/s，推理 FP16/BF16 .tensor core） |
| $W$ | 单卡出向带宽（字节/s，NVLink / PCIe） |
| $\tau$ | 单次 collective 固定延迟（秒/次；launch + 同步，经验 **10–50 μs**） |

**TP 推理前向（Megatron）：** 每层 **2 次** all-reduce（Attention 出口 + FFN 出口）。  
每层通信消息本体（FP16）：

$$
S_{\mathrm{TP,layer}} = 4\,B S D \quad\text{字节}.
$$

整步 Decode（$S=1$）collective **次数** $= 2L$。

---

# 第一部分：Prefill vs Decode 的计算强度 —— 为何 Decode 要排大 Batch

## 1.1 什么叫「计算强度」

这里用两个互补指标：

| 指标 | 定义 | 直觉 |
|------|------|------|
| **算力 $F$** | 一次 pass / step 的总 FLOPs（每 TP 卡） | 算得多不多 |
| **算术强度 $I$** | $I = F / M_{\mathrm{mem}}$（FLOPs / 激活字节访问量） | 每读 1 字节激活能算几次运算 |
| **有效并行度** | matmul 的 $m$ 维（Prefill：$B\!\cdot\!S$；Decode：$B$） | GPU 核能不能打满 |

Prefill **自带大 $S$** → $m = B\!\cdot\!S$ 很大；Decode **$S=1$** → $m = B$，**只靠 $B$ 撑并行度**。

---

## 1.2 每层算力公式（每 TP 卡）

**Prefill**（一次算完 prompt）：

$$
F_{\mathrm{prefill,layer}}
\approx
\underbrace{4 B S^2 D}_{\text{Attention }QK^\top,\,AV}
+
\underbrace{\frac{8 B S D^2}{N_{\mathrm{TP}}}}_{\text{QKV / 输出投影}}
+
\underbrace{\frac{6 B S D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}}_{\text{FFN}}.
$$

**Decode**（Cache 路径，每 step 1 个新 token / 序列）：

$$
F_{\mathrm{decode,layer}}
\approx
4 B S_{\mathrm{ctx}} D
+
\frac{8 B D^2}{N_{\mathrm{TP}}}
+
\frac{6 B D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
$$

**对 $B$ 的缩放：**

| 阶段 | $F$ 随 $B$ | 「胖」从哪来 |
|------|------------|--------------|
| Prefill | **线性** $\propto B$ | **$S$ 已在式子里**（$S^2$ 项） |
| Decode | **线性** $\propto B$ | **只有 $B$**；$S_{\mathrm{ctx}}$ 涨得再长也不增大 $m$ 维 |

---

## 1.3 算例：Llama 量级（$D=4096$，$D_{\mathrm{FF}}=14336$，$N_{\mathrm{TP}}=2$，$L=32$）

### Prefill：$B=1$，$S=2048$

```text
每层 FLOPs:
  Attention:  4·1·2048²·4096  +  8·1·2048·4096²/2  ≈  206 GFLOP
  FFN:        6·1·2048·4096·14336/2               ≈  361 GFLOP
  合计 ≈ 567 GFLOP/层  →  整 pass ≈ 18 TFLOP/卡

matmul 主维度:  m = B·S = 2048  （单请求已很胖）
```

设 $C = 150\ \mathrm{TFLOP/s}$：

$$
T_{\mathrm{compute,prefill}} \approx 18\ \mathrm{TFLOP} / 150\ \mathrm{TFLOP/s} \approx 120\ \mathrm{ms}.
$$

**Prefill 在 $B=1$ 时已接近算力饱和** —— prompt 长度 $S$ 充当「隐式 batch 维」。

### Decode：$B=1$，$S_{\mathrm{ctx}}=2048$

```text
每层 FLOPs:
  Attention:  4·1·2048·4096  +  8·4096²/2        ≈  101 MFLOP
  FFN:        6·1·4096·14336/2                     ≈  176 MFLOP
  合计 ≈ 277 MFLOP/层  →  整 step ≈ 8.9 GFLOP/卡

matmul 主维度:  m = B = 1  （极窄）
```

同样 $C = 150\ \mathrm{TFLOP/s}$，**峰值**算完只需：

$$
T_{\mathrm{peak}} \approx 8.9\ \mathrm{GFLOP} / 150\ \mathrm{TFLOP/s} \approx 0.06\ \mathrm{ms}.
$$

实际远慢于此：小矩阵 **有效 $C$** 常只有峰值的 **1–10%**；且 TP 还有 **固定延迟**（下节）。

### Decode：$B=32$（continuous batching）

```text
整 step FLOPs ≈ 8.9 GFLOP × 32 ≈ 285 GFLOP
matmul 主维度:  m = 32
```

**同一硬件上，Decode 要靠 $B$ 把 $m$ 从 1 拉到 32，才接近 Prefill 的「矩阵够胖、核够满」。**

---

## 1.4 固定延迟：Decode 为何特别怕 $B=1$

Decode 每 step 的 TP 通信（$S=1$）：

$$
M_{\mathrm{step}} = 4 B D L \quad\text{字节（消息本体）};\qquad N_{\mathrm{call}} = 2L.
$$

$B=1$，$L=32$，$D=4096$：$M_{\mathrm{step}} = 512\ \mathrm{KB}$，$N_{\mathrm{call}} = 64$。

| 腿 | $B=1$ | $B=32$ |
|----|-------|--------|
| 带宽时间 $M/W$（$W{=}100\ \mathrm{GB/s}$） | ≈ 0.005 ms | ≈ 0.16 ms |
| 延迟时间 $N_{\mathrm{call}}\cdot\tau$（$\tau{=}20\ \mu s$） | **≈ 1.3 ms** | **≈ 1.3 ms**（次数相同） |
| 算力 $F$（上例） | ≈ 8.9 GFLOP | ≈ 285 GFLOP |

**$B=1$ Decode：** 延迟 **≈ 1.3 ms** 与有效计算 **同阶**；带宽几乎空闲。  
**$B=32$ Decode：** 算力 **×32**，延迟 **不变** → 算力占比上升。

**Decode 临界 batch（延迟 vs 算力，粗估）：**

$$
\boxed{
B_{\mathrm{crit,lat}}
\;\approx\;
\frac{N_{\mathrm{call}}\,\tau}{T_{\mathrm{comp}}(B{=}1)}
\;\approx\;
\frac{2L\,\tau}{F_{\mathrm{step}}(B{=}1)/C_{\mathrm{eff}}}.
}
$$

代入 $L=32$，$\tau=20\ \mu s$，$F_{\mathrm{step}}(B{=}1)=8.9\ \mathrm{GFLOP}$，$C_{\mathrm{eff}}=30\ \mathrm{TFLOP/s}$（小矩阵有效算力）：

$$
T_{\mathrm{comp}}(1) \approx 0.3\ \mathrm{ms},\quad
B_{\mathrm{crit,lat}} \approx 1.28/0.3 \approx 4\text{–}20
\quad\text{（随 $\tau$、$C_{\mathrm{eff}}$ 浮动）}.
$$

工程上 **$B \gtrsim 16\text{–}32$** 是 continuous batching 的常见目标区。

---

## 1.5 对比总表：为何 Prefill 不等 Batch、Decode 要等 Batch

| | **Prefill** | **Decode** |
|--|-------------|------------|
| matmul 行维 | $B \cdot S$（$S$ 常 $\gg 1$） | $B$（$S=1$） |
| $B=1$ 时算力 | **18 TFLOP**（$S=2048$） | **8.9 GFLOP** |
| 固定 collective 次数 / 单位工作 | $2L$ **/ pass** | $2L$ **/ token** |
| 瓶颈（$B=1$） | 算力 + 带宽 | **延迟 + 窄 matmul** |
| 加大 $B$ 的作用 | 吞吐 $\uparrow$（多用户 prompt） | **把 $m$ 维撑胖 + 摊延迟** |
| 是否「必须」大 batch | **单用户 $S$ 已够** | **强依赖 continuous batching** |

**一句话：** Prefill 的「batch 维」在 **$S$**；Decode 的「batch 维」只能靠 **$B$**。

---

# 第二部分：Prefill / Decode 要不要 Tensor 并行 —— 量化判定

TP 做两件事：**① 切权重，让模型装进多卡；② 每层 2 次 all-reduce，付通信税。**

下面分 **必开 TP** 与 **通信是否划算** 两层说。

---

## 2.1 必开 TP：权重装不下

单卡权重（FP16，SwiGLU FFN，忽略 Embedding 粗算）：

$$
M_{\mathrm{weights}}
\approx
L \cdot \bigl(4 D^2 + 3 D D_{\mathrm{FF}}\bigr) \cdot 2\ \text{字节}.
$$

$D=4096$，$D_{\mathrm{FF}}=14336$，$L=32$：

$$
M_{\mathrm{weights}} \approx 32 \cdot (4\cdot4096^2 + 3\cdot4096\cdot14336) \cdot 2
\approx 32 \ \mathrm{GB}.
$$

**判定（硬条件）：**

$$
\boxed{
M_{\mathrm{weights}} > M_{\mathrm{GPU}}
\;\Longrightarrow\;
N_{\mathrm{TP}} \ge \left\lceil \frac{M_{\mathrm{weights}}}{M_{\mathrm{GPU}}} \right\rceil.
}
$$

Prefill 与 Decode **共用权重** → **两阶段适用性相同**。  
80 GB 卡、上例模型：**$N_{\mathrm{TP}}=1$ 可装**；70B 量级：**$N_{\mathrm{TP}} \ge 2\text{–}8$ 硬需求**。

---

## 2.2 通信是否压过算力：Prefill

每层 TP 通信时间（2 次环形 all-reduce，与 handout 一致）：

$$
T_{\mathrm{comm,layer}}
=
\frac{8\,(N_{\mathrm{TP}}-1)\,B S D}{N_{\mathrm{TP}}\,W}.
$$

每层算力时间：

$$
T_{\mathrm{comp,layer}}
=
\frac{4 B S^2 D + \dfrac{8 B S D^2 + 6 B S D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}}{C}.
$$

**长 prompt 时 Attention 的 $4BS^2D$ 主导。** 令 $T_{\mathrm{comm,layer}} \ge T_{\mathrm{comp,layer}}$ 且只保留主项（$B>0$ 约掉）：

$$
\frac{8(N_{\mathrm{TP}}-1)}{N_{\mathrm{TP}} W}
\ge
\frac{4 S^2 D}{C}
\quad\Longrightarrow\quad
\boxed{
S \;\le\;
S_{\mathrm{crit,prefill}}
=
\sqrt{\frac{2\,(N_{\mathrm{TP}}-1)\,C}{N_{\mathrm{TP}}\,W\,D}}.
}
$$

**读法：** prompt **短于** $S_{\mathrm{crit,prefill}}$ 时，TP 通信相对 **重**；**长于** 该值时，Prefill **算力主导**，TP **更划算**。

**数值（$N_{\mathrm{TP}}=2$，$C=150\ \mathrm{TFLOP/s}$，$W=100\ \mathrm{GB/s}$，$D=4096$）：**

$$
S_{\mathrm{crit,prefill}}
=
\sqrt{\frac{2 \cdot 1 \cdot 150\times10^{12}}{2 \cdot 100\times10^{9} \cdot 4096}}
\approx
\sqrt{36.6\times10^{3}}
\approx
192.
$$

| $S$ | Prefill 上 TP |
|-----|----------------|
| 512、2048、… | **算力主导**；$B=1$ 即可 |
| $\ll 192$（极短 prompt） | 通信用量相对显；仍可能需要 TP **装权重** |

**Prefill 对 $B$ 的阈值（通信视角）：** $B$ 在分子分母 **同阶约掉** → **Prefill 无 $B_{\mathrm{crit,TP}}$**；靠 **$S$** 养通信。

---

## 2.3 通信是否压过算力：Decode

每 **step**（生成 1 轮 token，$S=1$）：

$$
T_{\mathrm{comm,step}}
=
\frac{8\,(N_{\mathrm{TP}}-1)\,B D L}{N_{\mathrm{TP}}\,W},
\qquad
T_{\mathrm{comp,step}}
=
\frac{B L \bigl(4 S_{\mathrm{ctx}} D + \frac{8D^2 + 6D D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)}{C}.
$$

长上下文时 $4 S_{\mathrm{ctx}} D$ 主导。令 $T_{\mathrm{comm,step}} \ge T_{\mathrm{comp,step}}$（$B$ 约掉）：

$$
\boxed{
S_{\mathrm{ctx}} \;\le\;
S_{\mathrm{crit,decode,TP}}
=
\frac{2\,(N_{\mathrm{TP}}-1)\,C}{N_{\mathrm{TP}}\,W}.
}
$$

**同一组 $C,W,N_{\mathrm{TP}}$：**

$$
S_{\mathrm{crit,decode,TP}} \approx \frac{2 \cdot 1 \cdot 150\times10^{12}}{2 \cdot 100\times10^{9}}
\approx 1500\ \text{tokens}.
$$

| $S_{\mathrm{ctx}}$ | Decode 上 TP（通信 vs 算力） |
|--------------------|------------------------------|
| $\ll 1500$ | 通信 **相对重**（仍可能有 **装权重** 硬需求） |
| $\gtrsim 1500$ | 读 KV 算力 **渐增**，TP 通信 **相对轻** |

**Decode 对 $B$ 的阈值（与 §1.4 合并）：**

$$
\boxed{
B \;\ge\; B_{\mathrm{crit,decode}}
\;\approx\;
\max\!\left(
B_{\mathrm{crit,lat}},\;
B_{\mathrm{mem}}
\right).
}
$$

| 项 | 含义 | 量级（上例） |
|----|------|--------------|
| $B_{\mathrm{crit,lat}}$ | 摊 **$2L$ 次** collective 固定延迟 | **$\sim 16\text{–}32$** |
| $B_{\mathrm{mem}}$ | KV + 激活不超显存 | 随 $S_{\mathrm{ctx}}$、$L$ 变 |

**$B_{\mathrm{crit,decode}}$ 与 $S_{\mathrm{crit,decode,TP}}$ 分工：**

- **$S_{\mathrm{ctx}}$** 决定「读 cache 的算力够不够养 TP 通信」；
- **$B$** 决定「窄 matmul + 固定延迟下 GPU 够不够满」。

---

## 2.4 $N_{\mathrm{TP}}$ 开多大：通信临界点（Prefill 前向）

由 tensor-parallel 前向临界（单层 FFN+Attn 粗界，handout 同型）：

$$
T_{\mathrm{comm}} \ge T_{\mathrm{comp,fwd}}
\;\Longrightarrow\;
\boxed{
N_{\mathrm{TP}} \;\ge\;
1 + \frac{3}{2}\,D_{\mathrm{FF}}\,\frac{W}{C}.
}
$$

**数值（$D_{\mathrm{FF}}=14336$，$W/C = 1/1500$）：**

$$
N_{\mathrm{TP}} \;\ge\; 1 + \frac{3}{2}\cdot 14336 \cdot \frac{1}{1500} \approx 15.
$$

**读法：** 在 **Prefill 前向**、有效 $C/W \approx 1500$ 时，**$N_{\mathrm{TP}} \gtrsim 15$** 通信才开始压算力。  
实际 **8 卡 TP** 仍常见 —— 主因是 **装权重**，该式是 **「加 TP 卡是否还能线性加速 Prefill」** 的上界，与 **「要不要 TP」** 两回事。

**Decode** 有效 $C$ 更低 → 同一 $N_{\mathrm{TP}}$ 下 **Decode 更早撞通信墙** → 更依赖 **$B$**。

---

## 2.5 决策流程（推理 · 仅 TP）

```text
                    权重是否装进单卡？
                           │
              ┌────────────┴────────────┐
             否                         是
              │                         │
        开 TP，N_TP ≥ ⌈M_w/M_GPU⌉      可 N_TP = 1
              │                         │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
         Prefill 阶段              Decode 阶段
              │                         │
    S ≳ S_crit,prefill ?          B ≳ B_crit,decode ?
    （~200，上例）                 （~16–32，continuous batching）
              │                         │
    S 大 → TP 通信养得起          B 大 → 延迟摊薄、matmul 变胖
    S 小 → 通信占比↑              B 小 → 必须靠 batching 抬 B
              │                         │
              └────────────┬────────────┘
                           │
              两阶段共用同一 TP 组（权重一致）
              Prefill 靠 S；Decode 靠 B + S_ctx 读 cache
```

---

## 2.6 量化判定速查表

设 $C=150\ \mathrm{TFLOP/s}$，$W=100\ \mathrm{GB/s}$，$D=4096$，$D_{\mathrm{FF}}=14336$，$L=32$，$N_{\mathrm{TP}}=2$。

| 判定 | 公式 | 上例数值 | 含义 |
|------|------|----------|------|
| 必开 TP | $M_{\mathrm{weights}} > M_{\mathrm{GPU}}$ | 32 GB vs 卡容 | 装不下就要 TP |
| Prefill 算力主导 | $S \gtrsim S_{\mathrm{crit,prefill}}$ | $S \gtrsim 192$ | 短 prompt TP 通信相对显 |
| Decode 算力养 TP | $S_{\mathrm{ctx}} \gtrsim S_{\mathrm{crit,decode,TP}}$ | $S_{\mathrm{ctx}} \gtrsim 1500$ | 短 ctx 读 KV 少，通信相对显 |
| Decode 要 batch | $B \gtrsim B_{\mathrm{crit,decode}}$ | $B \gtrsim 16\text{–}32$ | continuous batching |
| Prefill 加 TP 卡收益递减 | $N_{\mathrm{TP}} \gtrsim 1+\frac{3}{2}D_{\mathrm{FF}}W/C$ | $\gtrsim 15$ | 通信压算力（非「要不要 TP」） |

---

## 2.7 结论：两阶段都适用 TP 吗？

| 问题 | 答案 |
|------|------|
| **权重装得下，还要 TP 吗？** | 单卡能装 → **$N_{\mathrm{TP}}=1$ 可行**；Decode 延迟更低 |
| **权重装不下** | **Prefill + Decode 都走同一 TP 组** |
| **Prefill 需要大 $B$ 吗？** | **通常不需要**；$S$ 提供「隐式 fat batch」 |
| **Decode 需要大 $B$ 吗？** | **需要**；$B_{\mathrm{crit,decode}} \sim 16\text{–}32$ 量级 |
| **TP 更适合哪段？** | **Prefill 更吃算力、更养得起 TP 通信**；Decode **更怕 $B=1$**，靠 batching 补救 |
| **实践** | **节点内 TP 两阶段共用**；优化 Decode → **continuous batching 抬 $B$**，Prefill 侧 **$S$ 自然够胖** |

---

## 附录：与「8 系数 / 4 次 AR」的对照

讲义 **训练全步** 每层 **4 次** all-reduce（前 2 + 反 2），系数 **8 = 4 × 环形 2 段**。  
**推理前向** 每层 **2 次** → 本文通信公式用 **$4BSD$ / 层**（FP16 消息本体），**$2L$ 次 / decode step**。

---

## 复习清单

1. Prefill $B=1,S=2048$ 为何够胖？ → $m=B\!\cdot\!S=2048$，整 pass **≈18 TFLOP**。  
2. Decode $B=1$ 为何瘦？ → $m=1$，**≈9 GFLOP/step** + **64 次** collective 延迟。  
3. $B_{\mathrm{crit,decode}}$ 解决什么？ → 摊 **$2L\tau$**，放大 matmul 行维。  
4. $S_{\mathrm{crit,prefill}}$ 解决什么？ → 短 prompt 时 TP 通信 **相对** 重。  
5. 何时必开 TP？ → $M_{\mathrm{weights}} > M_{\mathrm{GPU}}$，与阶段无关。
