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
| $H,\,d_k$ | Attention 头数、每头维（$D = H d_k$） |
| $N_{\mathrm{tok}}$ | matmul 展平后的 token 数（Prefill：$BS$；Decode：$B$） |
| $N_{\mathrm{TP}}$ | 张量并行度；**第一节固定为 1** |
| $F,\,M_{\mathrm{mem}}$ | 计算量（FLOP）、访存量（byte） |
| $C,\,C_{\mathrm{eff}}$ | 峰值 / 小矩阵有效算力 |
| $W$ | 单卡出向带宽（**仅第二节 TP 用**） |
| $\tau$ | 单次 kernel / collective 固定延迟（**10–50 μs**） |

---

# 第一节 · 无 TP（$N_{\mathrm{TP}}=1$）

单卡持有 **完整权重**；前向 **零跨卡通信**。瓶颈只有：**算力、matmul 形状、KV 读带宽、每层 kernel 启动**。

---

## 1.1 两个账本：计算 vs 访存

推理性能常看 **两个独立账本**：

| 账本 | 问什么 | 单位 | 决定什么 |
|------|--------|------|----------|
| **计算** | 一共做多少次浮点运算？ | FLOP | 算力 $C$（TFLOP/s）够不够 |
| **访存** | 一共从 HBM 读/写多少字节？ | byte | 带宽 $W_{\mathrm{mem}}$（GB/s）够不够 |

二者通过 **算术强度** $I = F / M_{\mathrm{mem}}$（FLOP/byte）连起来：$I$ 高 → 更偏 **计算瓶颈**；$I$ 低 → 更偏 **访存瓶颈**。

下面 **从一层 Transformer block 的每个 matmul 推起**，先 Prefill，再 Decode；最后再给汇总公式。

---

### 1.1.1 什么是「计算」？

**计算** = 前向里 **矩阵乘** 消耗的浮点运算次数（FLOP）。

**计数约定（与 handout / 本 repo 其它报告一致）：**

$$
(m,n)\cdot(n,p)\to(m,p)
\quad\Longrightarrow\quad
F_{\mathrm{matmul}} = 2mnp.
$$

- 每个输出元素：$n$ 次乘 + $(n{-}1)$ 次加 ≈ 按 **$2n$ FLOP/元素**，共 $m p$ 个输出 → **$2mnp$**。
- **逐元素** 运算（SiLU、LayerNorm、softmax 的 exp/div 等）在 **粗算** 里常 **忽略** 或单独一行；Prefill/Decode 的 **量级差异主要由 matmul 形状决定**。

**不算进本文主公式的东西：** 采样、embedding查表、CPU 调度、kernel launch——它们影响墙钟，但不进入下面的 $F_{\mathrm{layer}}$ 求和。

---

### 1.1.2 什么是「访存」？

**访存** = GPU **高带宽内存（HBM）** 上，前向一次 layer 需要 **读取 + 写入** 的 **激活与权重** 字节数（粗算，忽略 L2 命中、融合 kernel 等实现细节）。

对一次 matmul $Y = X W$（$X$ 为 $(m,n)$，$W$ 为 $(n,p)$），**至少** 要：

| 操作 | 元素数 | FP16 字节（×2） |
|------|--------|----------------|
| 读输入 $X$ | $mn$ | $2mn$ |
| 读权重 $W$ | $np$ | $2np$ |
| 写输出 $Y$ | $mp$ | $2mp$ |

$$
\boxed{
M_{\mathrm{matmul}} = 2\,(mn + np + mp)\ \text{字节}\quad(\text{FP16}).
}
$$

**一层 block 的总访存** ≈ 各 matmul 的 $M_{\mathrm{matmul}}$ 之和 + **KV cache 读写**（Decode 特有）。  
权重每层 **读一遍**；中间激活（Q/K/V、注意力矩阵、FFN 宽激活）**读+写** 各算一次。

**算术强度（单层、仅 matmul 粗算）：**

$$
I_{\mathrm{layer}} = \frac{F_{\mathrm{layer}}}{M_{\mathrm{layer}}}.
$$

---

### 1.1.3 一层 Block 里有什么？

标准 **Pre-LN** 一层（无 TP，单卡全权重）：

```text
输入 hidden  x : (B, S, D)
  │
  ├─ Attention
  │    ① QKV 线性：  x  →  Q, K, V
  │    ② SDPA：      QKᵀ → softmax → ·V
  │    ③ 输出投影：  attn_out → x'
  │
  └─ FFN（SwiGLU）
       ④ W₁： x' → h₁
       ⑤ W₂： x' → h₂ ；  gate = SiLU(h₁) ⊙ h₂
       ⑥ W₃： gate → 输出
```

**形状约定：** 凡线性层，把 $(B,S,D)$ **展平** 成 $(N_{\mathrm{tok}}, D)$，其中

$$
N_{\mathrm{tok}} = B \cdot S \quad\text{（Prefill）},\qquad
N_{\mathrm{tok}} = B \quad\text{（Decode，每 step 每序列 1 个新 token）}.
$$

Attention 内部用多头：$D = H \cdot d_k$（$H$ 头数，$d_k$ 每头维）。SDPA 的 FLOP 可合并为 **$4 N_{\mathrm{tok}} S_{\mathrm{attn}} D$**，其中 $S_{\mathrm{attn}}$ 是 **参与注意力乘法的序列长度**（Prefill 为 $S$；Decode 为 $S_{\mathrm{ctx}}$）。

---

### 1.1.4 Prefill · Attention：逐步推导

**设定：** $B$ 条序列，每条 prompt 长 $S$；一次算完所有 token。

#### 步骤 ① QKV 线性

$$
X \in \mathbb{R}^{(B S)\times D},\quad
W_{\mathrm{qkv}} \in \mathbb{R}^{D \times 3D}
\quad\Longrightarrow\quad
Q,K,V \text{ 各 } (B S, D).
$$

| | 计算 | 访存（FP16） |
|--|------|--------------|
| **FLOP** | $2 \cdot (BS) \cdot D \cdot (3D) = \mathbf{6\,B S D^2}$ | |
| **byte** | | $2\bigl(BSD + 3D^2 + 3BSD\bigr) \approx 8BSD + 6D^2$ |

（$D=4096$ 时 $D^2$ 项 $\sim 32\,\mathrm{MB/层}$，相对 $BSD$ 通常较小。）

#### 步骤 ② SDPA：$QK^\top$ 与 $\mathrm{softmax}(QK^\top)V$

reshape 为 $(B, H, S, d_k)$：

**$QK^\top$：** $(B,H,S,d_k) \cdot (B,H,d_k,S) \to (B,H,S,S)$

$$
F_{QK^\top} = 2 \cdot B \cdot H \cdot S \cdot d_k \cdot S
= 2 B S^2 D.
$$

**$\cdot V$：** $(B,H,S,S) \cdot (B,H,S,d_k) \to (B,H,S,d_k)$

$$
F_{\cdot V} = 2 \cdot B \cdot H \cdot S \cdot S \cdot d_k
= 2 B S^2 D.
$$

| | 计算 | 访存（主阶，FP16） |
|--|------|---------------------|
| **FLOP** | $F_{QK^\top} + F_{\cdot V} = \mathbf{4 B S^2 D}$ | |
| **byte** | 读 $Q,K,V$：$\sim 6BSD$；读写 $(B,H,S,S)$ 分数矩阵：$\sim 2BHS^2$ | $\sim 6BSD + 2BHS^2$ |

**Prefill 特征：** $S$ 大时 **$BHS^2$** 项（注意力矩阵）在访存里 **很显眼**——这是 Prefill 可能 **访存/算力双高** 的来源之一。

#### 步骤 ③ 输出投影

$$
(BS, D) \cdot (D, D) \to (BS, D)
\quad\Longrightarrow\quad
F = 2 \cdot BS \cdot D \cdot D = \mathbf{2 B S D^2}.
$$

| 访存 | $2(BSD + D^2 + BSD) \approx 4BSD + 2D^2$ |

#### Attention 小计（Prefill）

$$
\boxed{
F_{\mathrm{attn,prefill}} = 4 B S^2 D + 8 B S D^2.
}
$$

$$
M_{\mathrm{attn,prefill}} \approx
8 B S D + 2 B H S^2 + 8 D^2
\quad\text{（FP16，主阶）}.
$$

---

### 1.1.5 Prefill · FFN（SwiGLU）：逐步推导

三个线性层，中间宽 $D_{\mathrm{FF}}$：

| 步骤 | matmul 形状 $(m,n)\!\cdot\!(n,p)$ | FLOP | 访存主阶（FP16） |
|------|-------------------------------------|------|------------------|
| $W_1$ | $(BS,D)\!\cdot\!(D,D_{\mathrm{FF}})$ | $2BSDD_{\mathrm{FF}}$ | $2(BSD + DD_{\mathrm{FF}} + BSD_{\mathrm{FF}})$ |
| $W_2$ | 同上 | $2BSDD_{\mathrm{FF}}$ | 同上 |
| $W_3$ | $(BS,D_{\mathrm{FF}})\!\cdot\!(D_{\mathrm{FF}},D)$ | $2BSDD_{\mathrm{FF}}$ | $2(BSD_{\mathrm{FF}} + D_{\mathrm{FF}}D + BSD)$ |

逐元素 SiLU / 乘 **粗算忽略**。

$$
\boxed{
F_{\mathrm{ffn,prefill}} = 6 B S D D_{\mathrm{FF}}.
}
$$

$$
M_{\mathrm{ffn,prefill}} \approx
2\bigl(3 BSD + 2 BSD_{\mathrm{FF}} + DD_{\mathrm{FF}} + D_{\mathrm{FF}}D\bigr)
\approx 6 BSD + 4 BSD_{\mathrm{FF}} + 4 D D_{\mathrm{FF}}
\quad\text{（FP16）}.
$$

---

### 1.1.6 Prefill · 一层汇总

**计算（每层）：**

$$
\boxed{
F_{\mathrm{prefill,layer}}
=
\underbrace{4 B S^2 D}_{\text{SDPA}}
+
\underbrace{8 B S D^2}_{\text{QKV + 输出投影}}
+
\underbrace{6 B S D D_{\mathrm{FF}}}_{\text{FFN}}.
}
$$

**访存（每层，粗算）：**

$$
M_{\mathrm{prefill,layer}} \approx
8 B S D + 2 B H S^2 + 6 B S D + 4 B S D_{\mathrm{FF}} + \text{权重}
\approx
14 B S D + 2 B H S^2 + 4 B S D_{\mathrm{FF}} + 8 D^2 + 4 D D_{\mathrm{FF}}.
$$

**算术强度（看哪个子项主导）：**

| 子项 | FLOP 主阶 | 访存主阶 | 谁更「胖」 |
|------|-----------|----------|------------|
| SDPA | $4BS^2D$ | $2BHS^2$ | $S$ 大时 **FLOP $\propto S^2$** |
| 线性/FFN | $BS(D^2 + DD_{\mathrm{FF}})$ | $BSD + BSD_{\mathrm{FF}}$ | **FLOP/byte $\sim D$**，偏计算 |

Prefill 在 **$S$ 较大** 时：matmul 行维 $m = BS$ **已经很大** → **计算账本** 先饱和；SDPA 的 $S^2$ 项同时抬高 FLOP 与 $S^2$ 访存。

---

### 1.1.7 Decode 与 Prefill 差在哪？（KV cache）

Decode **每 step、每序列只算 1 个新 token**（$S_{\mathrm{new}}=1$），但 Attention 要 **读** 长度为 $S_{\mathrm{ctx}}$ 的 **KV cache**：

```text
Prefill：  Q, K, V  全部由当前  S  个 token 现算
          matmul 行维 N_tok = B·S

Decode：   Q  来自 1 个新 token
          K, V  从 cache 读  S_ctx  个历史 token（每层每头都要读）
          matmul 行维 N_tok = B  （不是 B·S_ctx！）
```

**关键：** $S_{\mathrm{ctx}}$ 进入 **Attention 的 $QK^\top$ / $\cdot V$**（FLOP $\propto S_{\mathrm{ctx}}$），但 **FFN / QKV 投影** 的 matmul 行维只有 **$B$**（每序列 1 token）。

---

### 1.1.8 Decode · Attention：逐步推导

**设定：** $B$ 条序列并行 decode；每条 **1 个新 token**；cache 长 $S_{\mathrm{ctx}}$。

#### 步骤 ① QKV（只对新 token）

$$
(B, D) \cdot (D, 3D)
\quad\Longrightarrow\quad
F = 2 \cdot B \cdot D \cdot 3D = \mathbf{6 B D^2}.
$$

#### 步骤 ② SDPA（$Q$ 长 1，$K,V$ 长 $S_{\mathrm{ctx}}$）

**$QK^\top$：** $(B,H,1,d_k) \cdot (B,H,d_k,S_{\mathrm{ctx}}) \to (B,H,1,S_{\mathrm{ctx}})$

$$
F_{QK^\top} = 2 \cdot B \cdot H \cdot 1 \cdot d_k \cdot S_{\mathrm{ctx}}
= 2 B S_{\mathrm{ctx}} D.
$$

**$\cdot V$：** $(B,H,1,S_{\mathrm{ctx}}) \cdot (B,H,S_{\mathrm{ctx}},d_k) \to (B,H,1,d_k)$

$$
F_{\cdot V} = 2 B S_{\mathrm{ctx}} D.
$$

**访存：** 除 matmul 外，**必须从 HBM 读 cache**：

$$
M_{\mathrm{KV,read}} \approx 2 \cdot 2 \cdot B S_{\mathrm{ctx}} D
= 4 B S_{\mathrm{ctx}} D \ \text{字节（FP16，K+V 各一份）}.
$$

（每层、每 step 读一次；$S_{\mathrm{ctx}}$ 越长，**访存 ∝ $S_{\mathrm{ctx}}$**，与 FLOP 同步涨。）

#### 步骤 ③ 输出投影

$$
F = 2 \cdot B \cdot D \cdot D = \mathbf{2 B D^2}.
$$

#### Attention 小计（Decode）

$$
\boxed{
F_{\mathrm{attn,decode}} = 4 B S_{\mathrm{ctx}} D + 8 B D^2.
}
$$

---

### 1.1.9 Decode · FFN：逐步推导

三个 matmul 的行维都是 **$m = B$**（不是 $B \cdot S_{\mathrm{ctx}}$）：

| 步骤 | FLOP |
|------|------|
| $W_1, W_2$ | 各 $2 B D D_{\mathrm{FF}}$ |
| $W_3$ | $2 B D D_{\mathrm{FF}}$ |

$$
\boxed{
F_{\mathrm{ffn,decode}} = 6 B D D_{\mathrm{FF}}.
}
$$

**访存主阶：** $6BD + 4BD_{\mathrm{FF}} + 4DD_{\mathrm{FF}}$（FP16）+ 权重 —— **与 $S_{\mathrm{ctx}}$ 无关**。

---

### 1.1.10 Decode · 一层汇总

**计算（每层，每 step）：**

$$
\boxed{
F_{\mathrm{decode,layer}}
=
\underbrace{4 B S_{\mathrm{ctx}} D}_{\text{读 cache 做 SDPA}}
+
\underbrace{8 B D^2}_{\text{QKV + 输出投影}}
+
\underbrace{6 B D D_{\mathrm{FF}}}_{\text{FFN}}.
}
$$

**访存（每层，每 step，粗算）：**

$$
M_{\mathrm{decode,layer}} \approx
\underbrace{4 B S_{\mathrm{ctx}} D}_{\text{读 KV cache}}
+
\underbrace{6 B D + 4 B D_{\mathrm{FF}}}_{\text{新 token 激活}}
+
8 D^2 + 4 D D_{\mathrm{FF}}
\quad\text{（+ 权重，同 Prefill）}.
$$

---

### 1.1.11 Prefill vs Decode：计算、访存、matmul 行维

| | **Prefill** | **Decode** |
|--|-------------|------------|
| **matmul 行维 $m$** | $B \cdot S$ | **$B$** |
| **Attention 序列维** | $S$（Q/K/V 同长） | **$S_{\mathrm{ctx}}$**（只读 cache） |
| **FLOP / 层** | $4BS^2D + 8BSD^2 + 6BSDD_{\mathrm{FF}}$ | $4BS_{\mathrm{ctx}}D + 8BD^2 + 6BDD_{\mathrm{FF}}$ |
| **谁提供「胖矩阵」** | **$S$** | **只有 $B$** |
| **访存随上下文** | $S^2$ 项（分数矩阵）+ $BS$ | **$S_{\mathrm{ctx}}$**（读 KV）+ $B$ |
| **固定 per token** | 一次 Prefill pass | **每 step 重复 $L$ 层** |

**一句话：** Prefill 的「batch 维」在 **$S$**（$m=B\!\cdot\!S$）；Decode 只能靠 **$B$** 把 FFN/QKV 的 matmul 撑胖 —— **与是否开 TP 无关**，是 **形状** 的结构性差异。

---

## 1.2 算例：$B=1$ 已经够胖 vs 必须靠 Batch

以下数字 **代入 §1.1 汇总公式**（无 TP，$D=4096$，$D_{\mathrm{FF}}=14336$，$L=32$）。

### Prefill：$B=1$，$S=2048$

```text
Attention:  4·2048²·4096  +  8·2048·4096²   ≈  275 GFLOP
FFN:        6·2048·4096·14336              ≈  722 GFLOP
合计 ≈ 997 GFLOP/层  →  整 pass ≈ 32 TFLOP

m = B·S = 2048
T_compute ≈ 32 TFLOP / 150 TFLOP/s ≈ 210 ms
```

**访存粗算（单层）：** $M \sim 14BSD + 2BHS^2 + 4BSD_{\mathrm{FF}} \approx 0.5\ \mathrm{GB}$ 量级（$S=2048$ 时 $S^2$ 项显著）。

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

**访存粗算（单层）：** 读 KV $\sim 4BS_{\mathrm{ctx}}D \approx 64\ \mathrm{MB}$；FFN/QKV 仅 $\sim B$ 项 → **算力极窄，但读 cache 仍随 $S_{\mathrm{ctx}}$ 线性涨**。

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
$$

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

1. **计算 vs 访存**：matmul FLOP 怎么数？访存三项是什么？ → $2mnp$；读 $X$、读 $W$、写 $Y$。  
2. **无 TP**：Prefill $m=$？Decode $m=$？ → $B\!\cdot\!S$ vs $B$。  
3. **无 TP**：$B=1,S=2048$ Prefill vs Decode 算力？ → **≈32 TFLOP** vs **≈17 GFLOP/step**。  
4. **Decode 访存**：谁随 $S_{\mathrm{ctx}}$ 涨？ → **读 KV cache**（$4BS_{\mathrm{ctx}}D$ / 层 / step）。  
5. **有 TP**：每层几次 AR？Decode 每 step 几次？ → **2 / 层**；**$2L$**（如 64）。  
6. **有 TP**：Prefill 仍不需大 $B$ 的原因？ → **$S$ 养算力**（与无 TP 相同）。  
7. **有 TP**：Decode 更要 batch 的原因？ → 第一节 $m=B$ **+** 摊 **$2L\tau$**。
