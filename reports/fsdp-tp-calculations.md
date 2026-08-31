# 二维并行（FSDP + TP）：从第一性原理推到底

> 对应 handout 的 `fsdp_tp_calcs`。在 [data-parallel-calculations.md](./data-parallel-calculations.md)、[fsdp-calculations.md](./fsdp-calculations.md)、[tensor-parallel-calculations.md](./tensor-parallel-calculations.md) 的基础上，把 **FSDP（怎么存）** 与 **TP（怎么算）** 叠成一张二维设备网格，推导前向 FLOP、通信时间、以及「最多能扩到多少张卡仍不被通信卡住」。

---

## 0. 核心结论

1. **FSDP** = Data Parallel + 参数 / 梯度 / optimizer state 分片；激活 $x$ 的窄维 $D$ 在 TP 组内保持完整。
2. **2D 并行** = 外层 **$N_{\mathrm{TP}}$ 个 TP 组**，每组内 **$N_{\mathrm{FSDP}}$ 路 FSDP**；8 卡即 2 组 × 每组 4 卡。
3. **FSDP 通信字节数** $S_{\mathrm{FSDP}} = 6DD_{\mathrm{FF}}/N_{\mathrm{TP}}$：all-gather 拼的是 **本 TP 组的子矩阵**；$N_{\mathrm{FSDP}}$ 进入环形系数 $(N_{\mathrm{FSDP}}-1)/N_{\mathrm{FSDP}}$。
4. **前向通信**：FSDP 横轴 all-gather 权重；TP 竖轴 all-reduce 激活。可重叠 → **max**；共享网络 → **相加**。

> 可视化：`misc/misc-fsdp-tp-ffn.html`（$D{=}4,\,D_{\mathrm{FF}}{=}8$，2 TP × 4 FSDP，卡名 R0C0…R1C3）。

---

## 0.5 切分一览：切什么、切哪一维

| 并行 | 切的对象 | 切的维度 | 切分类型 | 作用 |
|------|----------|----------|----------|------|
| **FSDP** | **激活** $x, y, dy, dx$ | batch 维 $B$ | 数据切分 | 每列 FSDP 卡只算 $B/N_{\mathrm{FSDP}}$ 个样本；$D$ 维保持完整 |
| **FSDP** | **权重** $W_1, W_2, W_3$ | $W_1/W_2$ 的行（$D$ 维）；$W_3$ 的列（$D$ 维） | **存储**切分 | 参数常驻 $1/N_{\mathrm{FSDP}}$；算前 all-gather 拼回 TP 子矩阵 |
| **FSDP** | **梯度** $dW$、**optimizer state** | 与权重同一维 | **存储**切分 | reduce-scatter 后每卡只留本地块 |
| **TP** | **权重** $W_1, W_2$ | 输出宽维 $D_{\mathrm{FF}}$（列） | **计算**切分 | 列并行：各 TP 组管宽维一块 |
| **TP** | **权重** $W_3$ | 输入宽维 $D_{\mathrm{FF}}$（行） | **计算**切分 | 行并行：与 $W_1/W_2$ 列切配对 |
| **TP** | **中间激活** $z$ 及宽维块 | $D_{\mathrm{FF}}$ | **计算**切分 | 宽激活留在本 TP 卡，不上网线 |
| **TP** | **输出激活** $y$、**输入梯度** $dx$ | 不切维；各 TP 卡算 partial | **计算**切分 | all-reduce 求和，每卡得完整 $(B/N_{\mathrm{FSDP}}, D)$ |

2D 并行 = 上表六行 **同时成立**：TP 定宽维怎么算，FSDP 定 batch + 权重/梯度/optimizer 怎么存。

### A. 激活 vs 权重（2D 下每卡长什么样）

| 对象 | 每卡形状 | 谁切、切什么 |
|------|----------|--------------|
| **激活** $x^{(j)}$ | $\bigl(\tfrac{B}{N_{\mathrm{FSDP}}},\, D\bigr)$ | FSDP **切激活的 batch 维**；$D$ 维完整，matmul 输入是完整窄向量 |
| **权重** $W_1^{(i,j)}$ | $\bigl(\tfrac{D}{N_{\mathrm{FSDP}}},\, \tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)$ | TP **切权重宽维列**（计算）；FSDP **切权重 $D$ 行**（存储）；gather 后得 $(D,\, D_{\mathrm{FF}}/N_{\mathrm{TP}})$ |
| **纯 TP 对激活** $x$ | $\tfrac{D}{N_{\mathrm{TP}}}$ 窄块 | 仅纯 TP：**切激活 $D$ 维**再 matmul；2D 下 **不切** $x$ 的 $D$ |

**FSDP 切 $D$ 只作用于权重/梯度/optimizer 的存储**；算 $xW$ 时 **激活** $x$ 的 $D$ 维在 TP 组内完整。2D 并行 = **TP 管权重宽维计算 + FSDP 管 batch 与权重存储**。

### B. 设备布局：2 个 TP 组，每组 4 张 FSDP 卡

```text
TP 组 R0（宽维 w0–w3）          TP 组 R1（宽维 w4–w7）
├── FSDP: R0C0, R0C1, R0C2, R0C3    ├── FSDP: R1C0, R1C1, R1C2, R1C3
└── 横排 4 卡 all-gather 权重        └── 横排 4 卡 all-gather 权重
         │                                      │
         └──── 竖向与 R1 组 all-reduce y ────────┘
              （同一 FSDP 列 Ck 内）
```

- **竖向（TP）**：TP 组之间各算宽维一半；**all-reduce 激活** $y$（切的是激活的 partial，宽维在本地）。
- **横向（FSDP）**：同一 TP 组内各存 **权重** 一小块；**all-gather / reduce-scatter 权重（及梯度）**。

### C. $S_{\mathrm{FSDP}}$ 为何含 $1/N_{\mathrm{TP}}$

- **$6DD_{\mathrm{FF}}$** = 全球三个权重矩阵总字节数（纯 FSDP 一次 gather 的目标大小）。
- **2D 下**每张卡只属于一个 TP 组，gather **本组半宽子矩阵** → 拼完大小 $\dfrac{6DD_{\mathrm{FF}}}{N_{\mathrm{TP}}}$。
- **$N_{\mathrm{FSDP}}$** 决定这 $\dfrac{6DD_{\mathrm{FF}}}{N_{\mathrm{TP}}}$ 拆成几份沿横排 gather → 环形系数 $(N_{\mathrm{FSDP}}-1)/N_{\mathrm{FSDP}}$。

---

## 符号表

| 符号 | 含义 |
|------|------|
| $B$ | 全局 batch（handout §8.5 用 $(B,D)$；若带序列长 $L$，下文所有 $B$ 可换为 $B\!\cdot\!L$，与 [tensor-parallel-calculations.md](./tensor-parallel-calculations.md) 一致） |
| $D$ | $d_{\mathrm{model}}$ |
| $D_{\mathrm{FF}}$ | FFN 中间维 |
| $N_{\mathrm{TP}}$ | 张量并行度：**切权重宽维 $D_{\mathrm{FF}}$（计算）** + **切激活 partial（all-reduce）** |
| $N_{\mathrm{FSDP}}$ | FSDP 并行度：**切 batch（激活）** + **切权重/梯度/optimizer 的 $D$ 维（存储）** |
| $N$ | 总设备数，$N = N_{\mathrm{TP}}\,N_{\mathrm{FSDP}}$ |
| $(i,j)$ | 设备坐标；下文也写 **TP 行 R$i$、FSDP 列 C$j$**，卡名 **R$i$C$j$**（见 HTML 可视化） |
| $C$ | 单设备算力（FLOP/s） |
| $W$ | 单设备出向带宽（字节/s） |
| $S_{\mathrm{FSDP}}$ | FSDP 轴上一次前向要 all-gather 的三个 **TP 子权重** 总字节数 |
| $S_{\mathrm{TP}}$ | TP 轴上一次前向 all-reduce 的 **激活** 字节数 |

---

## 1. 两个维度：嵌套结构，再画网格

### 1.1 一句话

- **TP（竖 / 外层 2）**：**切权重宽维** — $W_1,W_2$ 切 $D_{\mathrm{FF}}$ 列，$W_3$ 切 $D_{\mathrm{FF}}$ 行（计算切分）；组间 **all-reduce 激活** $y$。
- **FSDP（横 / 内层 4）**：**切 batch（激活）** + **切权重/梯度的 $D$ 维存储** — 在本 TP 子矩阵内按行/列再分 $N_{\mathrm{FSDP}}$ 份；组内 **all-gather 权重**。

两者组合：TP 定「全球 $W$ 的哪块宽维归我算」；FSDP 定「这块里 **权重存储** 的哪几行/列归我存」+「**激活 batch** 的哪几样本归我算」。

### 1.2 八卡例子：2 个 TP 组 × 每组 4 FSDP

**全局形状**（与 HTML 一致）：$D{=}4$（行 A B C D），$D_{\mathrm{FF}}{=}8$（列 w0–w7），$B{=}8$。

#### 嵌套（推荐记法）

```text
【TP 组 R0 · 负责 w0–w3】              【TP 组 R1 · 负责 w4–w7】
  R0C0  R0C1  R0C2  R0C3                  R1C0  R1C1  R1C2  R1C3
  └─ 4 张卡横排 all-gather W1⁽ᴿ⁰⁾ ─┘      └─ 4 张卡横排 all-gather W1⁽ᴿ¹⁾ ─┘
           │                                        │
           └──────── 竖排：R0Ck 与 R1Ck all-reduce y ─┘
                    （同一 FSDP 列 Ck）
```

#### 以 $W_1$ 为例（全球 $4\times 8$）

**TP 切权重宽维（计算）：** $W_1$ 按 **列** $D_{\mathrm{FF}}$ 切成两半，每 TP 组持一块。

```text
全球 W1:
  [ w0 w1 w2 w3 | w4 w5 w6 w7 ]
     TP 组 R0        TP 组 R1
     形状 4×4        形状 4×4   ← W1⁽ᴿ⁰⁾、W1⁽ᴿ¹⁾，各为半宽
```

**FSDP 切权重 $D$ 维（存储）：** 在 TP 组 R0 的 $4\times4$ 子矩阵内，按 **行** $D$ 分给横排 4 卡。

```text
W1⁽ᴿ⁰⁾ 的 4 行分给横排 4 卡:
  R0C0 存行 A (1×4)   R0C1 存行 B   R0C2 存行 C   R0C3 存行 D
```

**R0C2 常驻** $W_1^{(0,2)}$ 形状 $(1,\,4)$；**gather 横排后** 每张 R0C* 都有完整 $W_1^{(R0)}$ 形状 $(4,\,4)$。

**算 $xW_1$ 时：**

1. **FSDP 横排**（R0C0–R0C3）：all-gather → $W_1^{(R0)}$ 形状 $(4,\,4)$（本 TP 组半宽）。
2. **激活** $x^{(C2)}$ 形状 $(2,\,4)$：FSDP **切 batch 维**（$B/4{=}2$）；$D{=}4$ 完整。
3. **TP 竖排**（R0C2 与 R1C2）：各自 matmul 后 **all-reduce** $y$。

```text
         C0   C1   C2   C3   ← FSDP 横轴 · gather 权重
              ↓
R0 (TP)  [····][····][····][····]  宽 w0–w3
              ↕ all-reduce y
R1 (TP)  [····][····][····][····]  宽 w4–w7
```

### 1.3 三个权重的分片形状

| 权重 | TP 切（计算） | FSDP 切（存储） | 设备 $(i,j)$ 上常驻形状 |
|------|---------------|-----------------|-------------------------|
| $W_1,\,W_2$ | 宽维 $D_{\mathrm{FF}}$ 列 | 窄维 $D$ 行 | $\bigl(\tfrac{D}{N_{\mathrm{FSDP}}},\, \tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)$ |
| $W_3$ | 宽维 $D_{\mathrm{FF}}$ 行 | 窄维 $D$ 列 | $\bigl(\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}},\, \tfrac{D}{N_{\mathrm{FSDP}}}\bigr)$ |

**激活** $x^{(j)}$：FSDP **切 batch 维** → 形状 $\bigl(\tfrac{B}{N_{\mathrm{FSDP}}},\, D\bigr)$；$D$ 在 TP 组内完整复制。2D 下 batch 由 FSDP 切分（纯 TP 不切 batch）。

---

## 2. 前向：逐步写清「谁算什么、谁传什么」

设备 $(i,j)$ 上的前向（与 handout 一致）：

$$
\begin{aligned}
W_1^{(i)} &= \mathrm{all\text{-}gather}_{\ j}\bigl(\{W_1^{(i,j)}\}\bigr), &
W_2^{(i)} &= \mathrm{all\text{-}gather}_{\ j}\bigl(\{W_2^{(i,j)}\}\bigr), &
W_3^{(i)} &= \mathrm{all\text{-}gather}_{\ j}\bigl(\{W_3^{(i,j)}\}\bigr), \\[0.3em]
x_1^{(i,j)} &= x^{(j)}\,W_1^{(i)}, &
x_2^{(i,j)} &= x^{(j)}\,W_2^{(i)}, \\[0.3em]
z^{(i,j)} &= f\!\bigl(x_1^{(i,j)}\bigr)\odot x_2^{(i,j)}, \\[0.3em]
y^{(i,j)} &= z^{(i,j)}\,W_3^{(i)}, \\[0.3em]
y^{(j)} &= \mathrm{all\text{-}reduce}_{\ i}\bigl(\{y^{(i,j)}\}\bigr).
\end{aligned}
$$

读表：

| 步骤 | 计算 | FSDP 通信 | TP 通信 |
|------|------|-----------|---------|
| gather $W_1^{(i)},W_2^{(i)},W_3^{(i)}$ | — | **all-gather**（固定 $i$，沿 $j$） | — |
| $x_1,x_2,z,y^{(i,j)}$ | 与纯 TP 相同，但 batch 是 $B/N_{\mathrm{FSDP}}$ | — | — |
| 得到 $y^{(j)}$ | — | — | **all-reduce**（固定 $j$，沿 $i$） |

输出 **激活** $y^{(j)}$ 形状 $\bigl(\tfrac{B}{N_{\mathrm{FSDP}}},\, D\bigr)$：FSDP **切 batch 维**（每列 $Cj$ 持 $B/N_{\mathrm{FSDP}}$ 样本）；$D$ 维完整。

---

## 3. 反向传播

给定 $dy^{(j)}$（与 $y^{(j)}$ 同形）。TP 部分与 [tensor-parallel-calculations.md §3](./tensor-parallel-calculations.md) 相同，只是 $x,\,dy$ 的 batch 维是 $B/N_{\mathrm{FSDP}}$；FSDP 部分在权重梯度就绪后做 **reduce-scatter**（与 [fsdp-calculations.md §1](./fsdp-calculations.md) 相同，但对象是 **TP 子矩阵的梯度**）。

设备 $(i,j)$ 上：

$$
\begin{aligned}
dW_3^{(i,j)} &= \bigl(z^{(i,j)}\bigr)^\top dy^{(j)}, &
dz^{(i,j)} &= dy^{(j)}\,\bigl(W_3^{(i)}\bigr)^\top, \\[0.3em]
dx_2^{(i,j)} &= dz^{(i,j)}\odot f(x_1^{(i,j)}), &
dx_1^{(i,j)} &= dz^{(i,j)}\odot x_2^{(i,j)}\odot f'(x_1^{(i,j)}), \\[0.3em]
dW_1^{(i,j)} &= \bigl(x^{(j)}\bigr)^\top dx_1^{(i,j)}, &
dW_2^{(i,j)} &= \bigl(x^{(j)}\bigr)^\top dx_2^{(i,j)}, \\[0.3em]
dx^{(i,j)}_{\mathrm{local}} &= dx_1^{(i,j)}(W_1^{(i)})^\top + dx_2^{(i,j)}(W_2^{(i)})^\top, \\[0.3em]
dx^{(j)} &= \mathrm{all\text{-}reduce}_{\ i}\bigl(\{dx^{(i,j)}_{\mathrm{local}}\}\bigr), \\[0.3em]
dW_k^{(i,j)} &\leftarrow \mathrm{reduce\text{-}scatter}_{\ j}\bigl(\{dW_k^{(i,j)}\}\bigr), & k&=1,2,3.
\end{aligned}
$$

通信对称性：

| | FSDP 轴 | TP 轴 |
|--|---------|-------|
| 前向 | all-gather 权重（3 次） | all-reduce $y$（1 次） |
| 反向 | reduce-scatter $dW$（3 次） | all-reduce $dx_{\mathrm{local}}$（1 次） |

---

## 4. FLOP：为什么同时除以 $N_{\mathrm{TP}}$ 和 $N_{\mathrm{FSDP}}$

### 4.1 前向

gather 之后，设备 $(i,j)$ 做的是：

- $x^{(j)} W_1^{(i)}$：$\bigl(\tfrac{B}{N_{\mathrm{FSDP}}}, D\bigr)\cdot\bigl(D,\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)$ → FLOP $2\cdot\tfrac{B}{N_{\mathrm{FSDP}}}\cdot D\cdot\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$。
- $x^{(j)} W_2^{(i)}$：同上。
- $z^{(i,j)} W_3^{(i)}$：同上。

三个 matmul 相加：

$$
\boxed{
\text{每设备前向 FLOPs}
=
\frac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,N_{\mathrm{FSDP}}}.
}
$$

**两个 $N$ 从哪来：**

- $N_{\mathrm{FSDP}}$：FSDP **切激活 batch 维** → 每个 matmul 的 $m$ 维为 $B/N_{\mathrm{FSDP}}$。
- $N_{\mathrm{TP}}$：TP **切权重宽维** → 每个 matmul 的 $p$ 或收缩维为 $D_{\mathrm{FF}}/N_{\mathrm{TP}}$。

All-gather 是搬运，不计 FLOP。

### 4.2 反向

六个 matmul（与纯 TP / 纯 FSDP 相同计数），每个仍是最上面的 $2BD D_{\mathrm{FF}}/(N_{\mathrm{TP}} N_{\mathrm{FSDP}})$：

$$
\text{每设备反向 FLOPs}
=
\frac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,N_{\mathrm{FSDP}}}.
$$

---

## 5. 通信：两条轴，两种 collective

FP16，每元素 2 字节。环形 collective 系数见 [alternate-ring-all-reduce.md](./alternate-ring-all-reduce.md)。

### 5.1 FSDP 轴 — 传 **TP 子权重**

#### 纯 FSDP（无 TP）

全球 $W_1,W_2,W_3$ 合计 $6DD_{\mathrm{FF}}$ 字节。matmul 前 all-gather **完整** $W_k$，拼完大小 $D\times D_{\mathrm{FF}}$。

- **拼完大小** → $6DD_{\mathrm{FF}}$
- **gather 份数** → $N_{\mathrm{FSDP}}$ 份（环形系数 $(N_{\mathrm{FSDP}}-1)/N_{\mathrm{FSDP}}$）

#### 2D：gather 本 TP 组的 **权重** 子矩阵

TP **切权重 $W_1$ 的宽维列**（计算切分），全球 $W_1$ 分成 $N_{\mathrm{TP}}$ 块。R0 组 gather $W_1^{(R0)}$，形状 $(D,\, D_{\mathrm{FF}}/N_{\mathrm{TP}})$。

FSDP **切权重 $D$ 行存储**：横排把 $W_1^{(R0)}$ 从 $N_{\mathrm{FSDP}}$ 份行块拼回 $D$ 行完整。

三个 TP 子矩阵总字节数：

$$
\boxed{
S_{\mathrm{FSDP}}
=
\frac{6\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
}
$$

$N_{\mathrm{FSDP}}$ 进入环形 all-gather 的系数；$S_{\mathrm{FSDP}}$ 的分子只含 $N_{\mathrm{TP}}$：

$$
T_{\mathrm{comm}}^{\mathrm{FSDP,fwd}}
=
\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}
\cdot
\frac{6\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,W}.
$$

#### 数字实例（$D{=}4,\,D_{\mathrm{FF}}{=}8,\,N_{\mathrm{TP}}{=}2,\,N_{\mathrm{FSDP}}{=}4$）

| 量 | 纯 FSDP（无 TP） | 2D（本例） |
|----|------------------|------------|
| 全球 $W_1$ 元素 | $4\times 8 = 32$ | 32 |
| 一次 gather **拼完**的目标 | $4\times 8$，三矩阵共 **192 B** | TP 组内 $4\times 4$，三矩阵共 **96 B** $= 192/2$ |
| 每卡 **常驻** $W_1$ 块 | $1\times 8$（$1/4$ 行） | $1\times 4$（$1/4$ 行 × $1/2$ 宽） |
| $N_{\mathrm{FSDP}}$ 的作用 | 4 份行块沿横排 gather | **同上** — 仍是 4 份 |
| $N_{\mathrm{TP}}$ 的作用 | — | 只 gather **半宽** → 字节数 **÷2** |

**R0 横排 gather $W_1$：** R0C0…R0C3 各出 $1\times 4$ → 拼成 $4\times 4$（w0–w3）。**R1 组独立 gather** 自己的 $4\times 4$（w4–w7）。

记一句：**除以 $N_{\mathrm{TP}}$ =「你只拼自己 TP 组那半宽」；$N_{\mathrm{FSDP}}$ =「这半宽里拆了几份去借」。**

#### 公式

对固定 TP 行 $i$（或组 R$i$），all-gather 拼出 $W_1^{(i)},W_2^{(i)},W_3^{(i)}$：

$$
T_{\mathrm{comm}}^{\mathrm{FSDP,fwd}}
=
\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}
\cdot
\frac{6\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,W}.
$$

**$D_{\mathrm{FF}}$ 在分子、$N_{\mathrm{TP}}$ 在分母** — TP **切权重宽维**越细，FSDP all-gather 的 **权重** 字节数越小。FSDP 权重通信量不含 batch $B$。

### 5.2 TP 轴 — 传 **激活** $y$

对固定 FSDP 列 $j$（如 C2），all-reduce 形状 $\bigl(\tfrac{B}{N_{\mathrm{FSDP}}}, D\bigr)$ 的 FP16 张量：

$$
S_{\mathrm{TP}}
=
\frac{2\,B\,D}{N_{\mathrm{FSDP}}}.
$$

**同一列** $x^{(j)},y^{(j)}$ 在竖排两张 TP 卡（R0C$j$、R1C$j$）上 **batch 与 $D$ 相同**；TP 只汇总 partial $y$。

环形 all-reduce（两段）：

$$
T_{\mathrm{comm}}^{\mathrm{TP,fwd}}
=
\frac{2(N_{\mathrm{TP}}-1)}{N_{\mathrm{TP}}}
\cdot
\frac{S_{\mathrm{TP}}}{W}
=
\frac{4(N_{\mathrm{TP}}-1)\,B\,D}{N_{\mathrm{TP}}\,N_{\mathrm{FSDP}}\,W}.
$$

**数字实例（$B{=}8,\,D{=}4,\,N_{\mathrm{TP}}{=}2,\,N_{\mathrm{FSDP}}{=}4$）：** 列 C2 上 $y$ 形状 $(2,\,4)$ → $S_{\mathrm{TP}} = 2\times 8\times 4 / 4 = 16$ 字节；R0C2 与 R1C2 竖排 all-reduce 这 16 字节（环形系数再乘 $2(N_{\mathrm{TP}}-1)/N_{\mathrm{TP}}$）。

TP 轴传形状 $\bigl(\tfrac{B}{N_{\mathrm{FSDP}}}, D\bigr)$ 的窄激活；通信量含 $B$ 和 $D$，不含 $D_{\mathrm{FF}}$。

### 5.3 两轴通信的合并

两轴 collective **可并行**（不同 NIC / stream）→ 墙钟取 **max**：

$$
\boxed{
T_{\mathrm{comm,fwd}}
=
\max\!\left(
T_{\mathrm{comm}}^{\mathrm{FSDP,fwd}},\
T_{\mathrm{comm}}^{\mathrm{TP,fwd}}
\right).
}
$$

两轴 **共享同一网络、串行执行** → **相加**：

$$
T_{\mathrm{comm,fwd}}^{\mathrm{serial}}
=
T_{\mathrm{comm}}^{\mathrm{FSDP,fwd}}
+
T_{\mathrm{comm}}^{\mathrm{TP,fwd}}.
$$

反向同理：$\max$ 或 加 之间，一端是 reduce-scatter（FSDP），一端是 all-reduce（$dx$，TP）。

---

## 6. 最优拆分：两轴通信怎么平衡

固定总设备数 $N=N_{\mathrm{TP}}N_{\mathrm{FSDP}}$，可重叠时想 **minimize** $\max(T_{\mathrm{FSDP}}, T_{\mathrm{TP}})$。

近似 $(N-1)/N\approx 1$，$(N_{\mathrm{TP}}-1)/N_{\mathrm{TP}}\approx 1$，$(N_{\mathrm{FSDP}}-1)/N_{\mathrm{FSDP}}\approx 1$：

$$
T_{\mathrm{FSDP}}
\approx
\frac{6\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,W},
\qquad
T_{\mathrm{TP}}
\approx
\frac{4\,B\,D}{N_{\mathrm{FSDP}}\,W}
=
\frac{4\,B\,D\,N_{\mathrm{TP}}}{N\,W}.
$$

### 6.1 令 $T_{\mathrm{FSDP}}=T_{\mathrm{TP}}$

总通信墙钟 $\approx \max(T_{\mathrm{FSDP}}, T_{\mathrm{TP}})$。两轴相等时，给定 $N$ 下 $\max$ 取最小。

令 $T_{\mathrm{FSDP}}=T_{\mathrm{TP}}$：

$$
\frac{6\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}}
=
\frac{4\,B\,D\,N_{\mathrm{TP}}}{N}
\quad\Longrightarrow\quad
N_{\mathrm{TP}}^{2}
=
\frac{3\,D_{\mathrm{FF}}\,N}{2\,B}.
$$

故 **最优量级**（不必强行整数）：

$$
\boxed{
N_{\mathrm{TP}}^{\mathrm{opt}}
\approx
\sqrt{\frac{3\,D_{\mathrm{FF}}\,N}{2\,B}},
\qquad
N_{\mathrm{FSDP}}^{\mathrm{opt}}
=
\frac{N}{N_{\mathrm{TP}}^{\mathrm{opt}}}
\approx
\sqrt{\frac{2\,B\,N}{3\,D_{\mathrm{FF}}}}.
}
$$

读法：

- **batch 大** → $N_{\mathrm{FSDP}}^{\mathrm{opt}}$ 偏大（TP 轴通信含 $B$，要多 FSDP 列养得起）。
- **$D_{\mathrm{FF}}$ 大** → $N_{\mathrm{TP}}^{\mathrm{opt}}$ 偏大（FSDP 轴通信含 $D_{\mathrm{FF}}/N_{\mathrm{TP}}$，要多 TP 行把 gather 压小）。
- 总卡数增大时，$N_{\mathrm{TP}}^{\mathrm{opt}}$ 与 $N_{\mathrm{FSDP}}^{\mathrm{opt}}$ 同量级 $\sim \sqrt{N}$，两维同步开方。

### 6.2 数字实例：$N=8,\,B=8,\,D_{\mathrm{FF}}=8$（与本章小 FFN 同量级）

$$
N_{\mathrm{TP}}^{\mathrm{opt}}
\approx
\sqrt{\frac{3\times 8 \times 8}{2\times 8}}
=
\sqrt{12}
\approx 3.5
\;\Rightarrow\;
N_{\mathrm{TP}}\approx 3\text{–}4,\;
N_{\mathrm{FSDP}}\approx 2.
$$

Handout 常用 $N_{\mathrm{TP}}{=}2,\,N_{\mathrm{FSDP}}{=}4$ 作图例；最优拆分由公式算出，本例中 FSDP 轴通信约为 TP 轴的 3 倍。

代入 **$N_{\mathrm{TP}}=2,\,N_{\mathrm{FSDP}}=4,\,D=4,\,D_{\mathrm{FF}}=8,\,B=8$**（忽略环形 $(N{-}1)/N$）：

| 轴 | 粗算通信量级 | 谁主导 |
|----|--------------|--------|
| FSDP | $6DD_{\mathrm{FF}}/(N_{\mathrm{TP}}W) = 192/(2W) = 96/W$ | |
| TP | $4BD/(N_{\mathrm{FSDP}}W) = 128/(4W) = 32/W$ | **FSDP 轴 ≈ 3× 慢** |

此时增大 $N_{\mathrm{TP}}$ 或减小 $N_{\mathrm{FSDP}}$ 可使两轴更接近平衡。

---

## 7. 前向通信瓶颈：$N$ 最大能到多少

每设备计算时间：

$$
T_{\mathrm{compute,fwd}}
=
\frac{6\,B\,D\,D_{\mathrm{FF}}}{N\,C}.
$$

### 7.1 两轴可重叠（handout (c)）

最优拆分下 $T_{\mathrm{comm,fwd}}\approx T_{\mathrm{FSDP}}\approx T_{\mathrm{TP}}$。令 $T_{\mathrm{comm,fwd}}=T_{\mathrm{compute,fwd}}$：

$$
\frac{6\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,W}
=
\frac{6\,B\,D\,D_{\mathrm{FF}}}{N\,C}
\quad\Longrightarrow\quad
\frac{1}{N_{\mathrm{TP}}\,W}
=
\frac{B}{N\,C}
\quad\Longrightarrow\quad
N
=
\frac{B\,N_{\mathrm{TP}}\,C}{W}.
$$

代入 $N_{\mathrm{TP}}^{2}=3D_{\mathrm{FF}}N/(2B)$，解出临界 $N$：

$$
\boxed{
N
\;\ge\;
\frac{3}{2}\,B\,D_{\mathrm{FF}}\,\Bigl(\frac{W}{C}\Bigr)^{2}
\quad\Longrightarrow\quad
\text{通信开始压过计算（可重叠 + 最优拆分）}.
}
$$

**$D$ 被约掉**；瓶颈由 **batch × 宽层 × (带宽/算力)²** 决定。

### 7.2 两轴不可重叠（handout (d)）

$T_{\mathrm{comm,fwd}}^{\mathrm{serial}}=2\,T_{\mathrm{FSDP}}$（最优时两轴相等，**和** 是单侧两倍）。同样令等于 $T_{\mathrm{compute,fwd}}$：

$$
\frac{2}{N_{\mathrm{TP}}\,W}
=
\frac{B}{N\,C}
\quad\Longrightarrow\quad
N
=
\frac{B\,N_{\mathrm{TP}}\,C}{2\,W}.
$$

代入同一 $N_{\mathrm{TP}}^{2}$：

$$
\boxed{
N
\;\ge\;
\frac{3}{8}\,B\,D_{\mathrm{FF}}\,\Bigl(\frac{W}{C}\Bigr)^{2}
\quad\Longrightarrow\quad
\text{通信开始压过计算（串行 + 最优拆分）}.
}
$$

可重叠与串行两种设定下，临界 $N$ 相差约 4 倍：$(3/2)/(3/8)=4$。

---

## 8. 与「只开 TP / 只开 FSDP」的对照

| 策略 | 什么限制了加卡 | 临界量级（前向，粗记） |
|------|----------------|------------------------|
| 纯 FSDP | batch 太小 | $N_{\mathrm{FSDP}} \gtrsim 1 + B\,W/C$ |
| 纯 TP | $D_{\mathrm{FF}}$ 太小 | $N_{\mathrm{TP}} \gtrsim 1 + \tfrac{3}{2}D_{\mathrm{FF}}\,W/C$ |
| **FSDP + TP（2D）** | 两轴 **同时** 要养 | $N \gtrsim \tfrac{3}{2}B D_{\mathrm{FF}}(W/C)^2$（可重叠） |

纯 FSDP：通信 $\propto 6DD_{\mathrm{FF}}$，通信量不含 $B$。  
纯 TP：通信 $\propto BLD$，计算 $\propto BLD D_{\mathrm{FF}}$，临界 $\propto D_{\mathrm{FF}}$。

2D 把 batch 维与 $D_{\mathrm{FF}}$ 维的扩展能力相乘：$N_{\mathrm{crit}} \propto B\cdot D_{\mathrm{FF}}$。工程上常 **TP 放节点内、FSDP 跨节点**，两轴各负责一个可扩展维度。

Critical batch size 与 scaling laws 指出：单靠加大 batch 喂 FSDP 会触优化上限；2D 并行还可沿 **$D_{\mathrm{FF}}$ / TP** 轴继续扩 $N$。

---

## 9. 数值直觉（XL 档，$C/W\approx 4000$）

取 $D_{\mathrm{FF}}=10240$，$B=4000$，$C/W = 1/4000$：

$$
N_{\mathrm{crit,overlap}}
\approx
\frac{3}{2}\times 4000 \times 10240 \times \frac{1}{4000^{2}}
\approx
\frac{3}{2}\times\frac{10240}{4000}
\approx 3.8.
$$

在这组 **偏小 batch** 的设定下，即使用最优 2D 拆分，总设备数 $N$ 也就 **3–4 卡** 量级就会通信主导——和单层 FFN 上「PCIe + 小 batch」的悲观结论一致。真实预训练把 $B$ 提到 $10^{5}$ 量级，$N_{\mathrm{crit}}$ 会按 $B$ 线性抬高。

---

## 10. 公式汇总

符号：$N=N_{\mathrm{TP}}N_{\mathrm{FSDP}}$；$S_{\mathrm{FSDP}}=6DD_{\mathrm{FF}}/N_{\mathrm{TP}}$；$S_{\mathrm{TP}}=2BD/N_{\mathrm{FSDP}}$；环形 collective。

| | 纯 TP | 纯 FSDP | FSDP + TP（2D） |
|--|-------|---------|-----------------|
| **每设备前向 FLOPs** | $\dfrac{6BD D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$ | $\dfrac{6BD D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$ | $\dfrac{6BD D_{\mathrm{FF}}}{N_{\mathrm{TP}}N_{\mathrm{FSDP}}}$ |
| **每设备反向 FLOPs** | $\dfrac{12BD D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$ | $\dfrac{12BD D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$ | $\dfrac{12BD D_{\mathrm{FF}}}{N_{\mathrm{TP}}N_{\mathrm{FSDP}}}$ |
| **前向 FSDP 轴通信** | — | $\dfrac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}\dfrac{S}{W}$ | $\dfrac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}\dfrac{S_{\mathrm{FSDP}}}{W}$ |
| **前向 TP 轴通信** | $\dfrac{2(N_{\mathrm{TP}}-1)}{N_{\mathrm{TP}}}\dfrac{2BD}{W}$ | — | $\dfrac{2(N_{\mathrm{TP}}-1)}{N_{\mathrm{TP}}}\dfrac{S_{\mathrm{TP}}}{W}$ |
| **前向总通信（可重叠）** | （仅 TP 项） | （仅 FSDP 项） | $\max(T_{\mathrm{FSDP,fwd}},\, T_{\mathrm{TP,fwd}})$ |
| **前向总通信（不可重叠）** | 同左 | 同左 | $T_{\mathrm{FSDP,fwd}}+T_{\mathrm{TP,fwd}}$ |
| **最优拆分（$N$ 固定）** | — | — | $N_{\mathrm{TP}}\approx\sqrt{3D_{\mathrm{FF}}N/(2B)}$ |
| **前向通信瓶颈（最优，可重叠）** | $N_{\mathrm{TP}}\gtrsim 1+\tfrac{3}{2}D_{\mathrm{FF}}W/C$ | $N_{\mathrm{FSDP}}\gtrsim 1+BW/C$ | $N\gtrsim \tfrac{3}{2}BD_{\mathrm{FF}}(W/C)^{2}$ |
| **前向通信瓶颈（最优，不可重叠）** | — | — | $N\gtrsim \tfrac{3}{8}BD_{\mathrm{FF}}(W/C)^{2}$ |

（纯 TP / FSDP 单独行的公式见 [tensor-parallel-calculations.md](./tensor-parallel-calculations.md)、[fsdp-calculations.md](./fsdp-calculations.md)；上表 $S=6DD_{\mathrm{FF}}$ 为 FSDP 单层全权重字节数。）
