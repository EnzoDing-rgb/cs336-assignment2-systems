# 张量并行（Tensor Parallel）算清楚：以单个 FFN 层为例

> 对应 handout 的 `tp_calcs`。设定与 [data-parallel-calculations.md](./data-parallel-calculations.md) 相同：单设备算力 $C$（FLOP/s）、出向带宽 $W$（字节/s）、激活与梯度均为 FP16（每元素 2 字节）。这里不再切 batch / 序列，而是把 **权重矩阵的某一维** 切到 $N_{\mathrm{TP}}$ 台设备上。
>
> **形状记号：** 真实 Transformer 里 FFN 输入是 $(B, L, D)$（batch × 序列长 × 隐藏维）。矩阵乘时把前两维展平成 **token 数** $B\!\cdot\!L$，即按 $(B\!\cdot\!L,\, D)$ 与权重相乘——与代码里 `(batch, context, d_model)` reshape 后做 Linear 一致。下文公式里的 $B\!\cdot\!L$ 就是这个展平后的 leading dimension；$S$ 保留给 **通信字节数**（不是 sequence length），故序列长度用大写 $L$。

数据并行是「每台拿完整模型、不同样本」。张量并行是「每台拿同一批样本、但只拿权重的一块」。对 FFN 来说，关键问题只有三个：

1. 切完之后，前向 / 反向每台到底算什么？
2. 哪里必须通信，传多大？
3. $N_{\mathrm{TP}}$ 大到什么程度，通信时间会压过计算时间？

下面从「一个 matmul 怎么切」讲起，再拼成完整 FFN，最后把 FLOP、通信、瓶颈临界点一次算完。

---

## 符号表

| 符号 | 含义 |
|------|------|
| $B$ | batch（样本数；TP **不切** batch，每台都看完整 $B$） |
| $L$ | 序列长度 sequence length（代码里常写 `seq_len` / `context`；TP **不切** $L$） |
| $D$ | 隐藏维度 $d_{\mathrm{model}}$ |
| $D_{\mathrm{FF}}$ | FFN 中间维度 |
| $N_{\mathrm{TP}}$ | 张量并行设备数 |
| $C$ | 单设备算力（FLOP/s） |
| $W$ | 单设备出向带宽（字节/s） |
| $S$ | 一次 all-reduce 的 **消息字节数**（$S = 2BLD$；**不是** sequence length） |
| $W_1, W_2$ | 升维权重，完整形状 $(D, D_{\mathrm{FF}})$ |
| $W_3$ | 降维权重，完整形状 $(D_{\mathrm{FF}}, D)$ |
| $W_1^{(i)}, W_2^{(i)}$ | 设备 $i$ 上的列并行分片，形状 $\bigl(D,\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)$ |
| $W_3^{(i)}$ | 设备 $i$ 上的行并行分片，形状 $\bigl(\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}, D\bigr)$ |
| $x$ | 输入，逻辑形状 $(B, L, D)$；矩阵乘时展平为 $(B\!\cdot\!L,\, D)$；**每台都有完整的一份** |
| $x_1^{(i)}, x_2^{(i)}, z^{(i)}$ | 设备 $i$ 上的中间激活，形状 $\bigl(B\!\cdot\!L,\,\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)$ |
| $y^{(i)}$ | 设备 $i$ 的局部输出，形状 $(B\!\cdot\!L,\, D)$ |
| $y$ | all-reduce 之后的完整输出，形状 $(B\!\cdot\!L,\, D)$（等价于 $(B, L, D)$） |
| $f$ | 逐元素激活（如 SiLU）；$f'$ 为其导数 |
| $\cdot$ | 矩阵乘；$\odot$ 逐元素乘；${}^\top$ 转置 |

矩阵乘代价约定不变：形状 $(m,n)\cdot(n,p)\to(m,p)$ 需要 $2mnp$ 次 FLOP。逐元素运算（$f$、$f'$、$\odot$）在本课计数里忽略。

---

## 1. 先搞清楚：一个 matmul 的两种切法

目标：$y = xW$，其中 $x$ 形状 $(B\!\cdot\!L,\, D)$（即 $(B,L,D)$ 展平后），$W$ 形状 $(D, D_{\mathrm{FF}})$。

### 1.1 列并行（column parallel）：切输出维

把 $W$ 按 **列** 切开：

$$
W = \bigl[W^{(0)} \mid W^{(1)} \mid \cdots \mid W^{(N_{\mathrm{TP}}-1)}\bigr],
\qquad
W^{(i)}\ \text{形状}\ \Bigl(D,\ \frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\Bigr).
$$

每台用 **完整** $x$ 乘自己那一列块：

$$
y^{(i)} = x\,W^{(i)}
\qquad\text{形状}\ \Bigl(B\!\cdot\!L,\ \tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\Bigr).
$$

各台得到的是输出的不同列块。若后面某步需要完整 $y$，就做一次 **all-gather**，把各列拼回去：

$$
y = \mathrm{all\text{-}gather}\bigl(\{y^{(i)}\}\bigr).
$$

直觉：每台负责输出的一段「宽度」。

### 1.2 行并行（row parallel）：切输入维

把 $W$ 按 **行** 切开：

$$
W =
\begin{bmatrix}
W^{(0)} \\
\vdots \\
W^{(N_{\mathrm{TP}}-1)}
\end{bmatrix},
\qquad
W^{(i)}\ \text{形状}\ \Bigl(\tfrac{D}{N_{\mathrm{TP}}},\ D_{\mathrm{FF}}\Bigr).
$$

同时把 $x$ 按列切成 $x^{(i)}$，形状 $\bigl(B\!\cdot\!L,\,\tfrac{D}{N_{\mathrm{TP}}}\bigr)$，然后

$$
y^{(i)} = x^{(i)}\,W^{(i)}
\qquad\text{形状}\ (B\!\cdot\!L,\, D_{\mathrm{FF}}).
$$

这里每台算出的都是「对完整输出的一份 **部分和**」。因为

$$
xW = \sum_{i} x^{(i)} W^{(i)},
$$

所以要用 **all-reduce（求和）** 把各台的 $y^{(i)}$ 加起来：

$$
y = \mathrm{all\text{-}reduce}\bigl(\{y^{(i)}\}\bigr).
$$

直觉：每台负责输入的一段「宽度」，输出维度完整，但数值只是部分贡献。

### 1.3 为什么 FFN 要「先列后行」配对？

FFN 的结构是：

```text
升维（变宽） → 逐元素非线性 → 降维（变窄回 D）
```

若升维用 **列并行**，每台手里已经是宽维的一块 $\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$；降维再对这块做 **行并行**，输入维恰好对上，**中间不必 all-gather**。整条 FFN 只需在降维之后做 **一次 all-reduce**。

这就是 Megatron 风格 FFN 张量并行的核心：用切分方式的配对，把两次通信压成一次。

---

## 2. 本题的前向：写清楚每一步在算什么

给定完整输入 $x\in\mathbb{R}^{B\times L\times D}$（矩阵乘时视为 $\mathbb{R}^{(B\cdot L)\times D}$）。设备 $i$ 持有：

- $W_1^{(i)}, W_2^{(i)}\in\mathbb{R}^{D \times (D_{\mathrm{FF}}/N_{\mathrm{TP}})}$（列并行）
- $W_3^{(i)}\in\mathbb{R}^{(D_{\mathrm{FF}}/N_{\mathrm{TP}}) \times D}$（行并行）

前向：

$$
\begin{aligned}
x_1^{(i)} &= x\,W_1^{(i)}, \\
x_2^{(i)} &= x\,W_2^{(i)}, \\
z^{(i)} &= f\!\bigl(x_1^{(i)}\bigr)\odot x_2^{(i)}, \\
y^{(i)} &= z^{(i)}\,W_3^{(i)}, \\
y &= \mathrm{all\text{-}reduce}\bigl(\{y^{(i)}\}_{i=0}^{N_{\mathrm{TP}}-1}\bigr).
\end{aligned}
$$

读法：

| 步骤 | 每台在做什么 | 要通信吗 |
|------|--------------|----------|
| $x_1^{(i)}, x_2^{(i)}$ | 用完整 $x$ 乘自己的列分片，得到宽维的一块 | 否 |
| $z^{(i)}$ | 只在自己那一块上做逐元素运算 | 否 |
| $y^{(i)}$ | 用自己的行分片把宽维块压回 $(B\!\cdot\!L,\, D)$，得到部分和 | 否 |
| $y$ | 把各台部分和加总，每台都拿到完整输出 | **是：all-reduce** |

前向里 **唯一** 的集合通信，就是最后这一次对形状 $(B\!\cdot\!L,\, D)$ 张量的 all-reduce。

---

## 3. 反向：从「单个 matmul 的梯度公式」推到分片版

### 3.1 先复习：未分片 FFN 的反向（对照 §8.2）

未分片时，前向是 $x_1=xW_1,\ x_2=xW_2,\ z=f(x_1)\odot x_2,\ y=zW_3$。给定 $dy$（形状 $(B\!\cdot\!L,\, D)$）：

$$
\begin{aligned}
dz &= dy\,W_3^\top, \\
dx_2 &= dz \odot f(x_1), \\
dx_1 &= dz \odot x_2 \odot f'(x_1), \\
dW_3 &= z^\top\,dy, \\
dW_1 &= x^\top\,dx_1, \\
dW_2 &= x^\top\,dx_2, \\
dx &= dx_1\,W_1^\top + dx_2\,W_2^\top.
\end{aligned}
$$

规律只有一条：**每个前向 matmul，反向对应「对输入的梯度」和「对权重的梯度」各一次 matmul**。三个权重 → 反向六个 matmul。

### 3.2 分片之后：通信落在哪里？

handout 假定每台都拿到同一份 $dy\in\mathbb{R}^{(B\cdot L)\times D}$（前向 all-reduce 之后每台都有完整 $y$，后续损失对 $y$ 的梯度在各台上一致）。

因为前向 $y=\sum_i y^{(i)}$，所以

$$
\frac{\partial L}{\partial y^{(i)}} = \frac{\partial L}{\partial y} = dy.
$$

也就是说：**反向进入 $W_3$ 这一步时，每台直接用同一份 $dy$，这里不再额外通信。**

于是设备 $i$ 上：

**（1）行并行的 $W_3^{(i)}$**

$$
\begin{aligned}
dW_3^{(i)} &= \bigl(z^{(i)}\bigr)^\top\, dy, \\
dz^{(i)} &= dy\,\bigl(W_3^{(i)}\bigr)^\top.
\end{aligned}
$$

$dz^{(i)}$ 形状 $\bigl(B\!\cdot\!L,\,\tfrac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\bigr)$，正好只覆盖本台负责的那一段宽维。

**（2）逐元素门控**

$$
\begin{aligned}
dx_2^{(i)} &= dz^{(i)} \odot f\!\bigl(x_1^{(i)}\bigr), \\
dx_1^{(i)} &= dz^{(i)} \odot x_2^{(i)} \odot f'\!\bigl(x_1^{(i)}\bigr).
\end{aligned}
$$

**（3）列并行的 $W_1^{(i)}, W_2^{(i)}$**

$$
\begin{aligned}
dW_1^{(i)} &= x^\top\, dx_1^{(i)}, \\
dW_2^{(i)} &= x^\top\, dx_2^{(i)}.
\end{aligned}
$$

对输入 $x$ 的贡献，本台只能给出 **一部分**：

$$
dx^{(i)}_{\mathrm{local}}
=
dx_1^{(i)}\,\bigl(W_1^{(i)}\bigr)^\top
+
dx_2^{(i)}\,\bigl(W_2^{(i)}\bigr)^\top.
$$

完整的 $dx$ 是所有列分片贡献之和：

$$
dx
=
\sum_{i=0}^{N_{\mathrm{TP}}-1}
dx^{(i)}_{\mathrm{local}}
=
\mathrm{all\text{-}reduce}\bigl(\{dx^{(i)}_{\mathrm{local}}\}\bigr).
$$

### 3.3 反向方程汇总（本题 (a)）

设备 $i$ 上，给定 $dy$，并使用前向保存的 $x,\ x_1^{(i)},\ x_2^{(i)},\ z^{(i)}$：

$$
\begin{aligned}
dW_3^{(i)}
&=
\bigl(z^{(i)}\bigr)^\top\, dy, \\[0.4em]
dz^{(i)}
&=
dy\,\bigl(W_3^{(i)}\bigr)^\top, \\[0.4em]
dx_2^{(i)}
&=
dz^{(i)} \odot f\!\bigl(x_1^{(i)}\bigr), \\[0.4em]
dx_1^{(i)}
&=
dz^{(i)} \odot x_2^{(i)} \odot f'\!\bigl(x_1^{(i)}\bigr), \\[0.4em]
dW_1^{(i)}
&=
x^\top\, dx_1^{(i)}, \\[0.4em]
dW_2^{(i)}
&=
x^\top\, dx_2^{(i)}, \\[0.4em]
dx^{(i)}_{\mathrm{local}}
&=
dx_1^{(i)}\,\bigl(W_1^{(i)}\bigr)^\top
+
dx_2^{(i)}\,\bigl(W_2^{(i)}\bigr)^\top, \\[0.4em]
dx
&=
\mathrm{all\text{-}reduce}\bigl(\{dx^{(i)}_{\mathrm{local}}\}_{i=0}^{N_{\mathrm{TP}}-1}\bigr).
\end{aligned}
$$

产物：每台得到自己的 $dW_1^{(i)}, dW_2^{(i)}, dW_3^{(i)}$，以及所有设备上一致的完整 $dx$。

**前后向通信的对称性**（务必记住）：

| | 集合通信 | 张量形状 |
|--|----------|----------|
| 前向 | 1 次 all-reduce（汇总 $y^{(i)}$） | $(B\!\cdot\!L,\, D)$ |
| 反向 | 1 次 all-reduce（汇总 $dx^{(i)}_{\mathrm{local}}$） | $(B\!\cdot\!L,\, D)$ |

列并行「前向安静、反向要汇总输入梯度」；行并行「前向要汇总输出、反向安静」。两者配对之后，整条 FFN 前、反向各付一次 all-reduce。

---

## 4. FLOP 怎么数（本题 (b)）

### 4.1 前向

每台三个 matmul，中间宽维都是 $D_{\mathrm{FF}}/N_{\mathrm{TP}}$：

$$
\begin{aligned}
x\,W_1^{(i)} &: \quad 2\,B\,L\,D\,\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}, \\
x\,W_2^{(i)} &: \quad 2\,B\,L\,D\,\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}, \\
z^{(i)}\,W_3^{(i)} &: \quad 2\,B\,L\,\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}\,D
= 2\,B\,L\,D\,\frac{D_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
\end{aligned}
$$

因此 **每设备前向 FLOPs**：

$$
\frac{6\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
$$

一句话：未分片前向是 $6BLD D_{\mathrm{FF}}$；TP 把每个 matmul 的宽维（或等价的收缩维）缩成 $1/N_{\mathrm{TP}}$，故每台计算量除以 $N_{\mathrm{TP}}$。

### 4.2 反向

仍是六个 matmul（三个权重梯度 + 三个激活侧梯度），每个代价同为 $2\,B\,L\,D\,D_{\mathrm{FF}}/N_{\mathrm{TP}}$：

$$
\frac{12\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}}.
$$

一句话：反向 matmul 数是前向的两倍，每个仍按分片后的 $D_{\mathrm{FF}}/N_{\mathrm{TP}}$ 计，故为前向的两倍。

对比数据并行：DP 是用 $B/N_{\mathrm{DP}}$ 除以设备数；TP 是用 $D_{\mathrm{FF}}/N_{\mathrm{TP}}$ 除以设备数。两者都把每台 FLOP 降到原来的 $1/N$，但切的轴不同。$L$ 在 DP 与 TP 里都不切，只作为 token 数 $BL$ 的一部分进入 FLOP 与通信量。

---

## 5. 通信时间怎么数（本题 (c)）

通信对象始终是形状 $(B\!\cdot\!L,\, D)$ 的 FP16 张量，字节数：

$$
S = 2\,B\,L\,D.
$$

环形 all-reduce（先 reduce-scatter 再 all-gather；推导见 [alternate-ring-all-reduce.md](./alternate-ring-all-reduce.md)）下，每设备出向量为 $2\cdot\tfrac{N_{\mathrm{TP}}-1}{N_{\mathrm{TP}}}\,S$，故

$$
T_{\mathrm{comm}}
=
\frac{2(N_{\mathrm{TP}}-1)}{N_{\mathrm{TP}}}\cdot\frac{2BLD}{W}
=
\frac{4(N_{\mathrm{TP}}-1)\,BLD}{N_{\mathrm{TP}}\,W}.
$$

- **前向**：一次 all-reduce → 上式。
- **反向**：同样一次、同样大小 → **同一式**。

一句话（前向）：前向只需对 $(B\!\cdot\!L,\, D)$ 的输出做一次环形 all-reduce，耗时 $\frac{4(N_{\mathrm{TP}}-1)BLD}{N_{\mathrm{TP}} W}$。  
一句话（反向）：反向只需对 $(B\!\cdot\!L,\, D)$ 的输入梯度做一次同样的 all-reduce，耗时相同。

注意：$D_{\mathrm{FF}}$ **不进**通信量——宽维分片留在各台本地，跨设备传的永远是窄的 $(B\!\cdot\!L,\, D)$（即逻辑形状 $(B,L,D)$）。

---

## 6. 什么时候通信压过计算（本题 (d)）

计算时间（每设备）：

$$
\begin{aligned}
T_{\mathrm{compute}}^{\mathrm{fwd}}
&=
\frac{6\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,C}, \\[0.4em]
T_{\mathrm{compute}}^{\mathrm{bwd}}
&=
\frac{12\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,C}.
\end{aligned}
$$

通信时间（前、反向相同）：

$$
T_{\mathrm{comm}}
=
\frac{4(N_{\mathrm{TP}}-1)\,BLD}{N_{\mathrm{TP}}\,W}.
$$

约定：计算与通信可重叠时，若 $T_{\mathrm{comm}}\ge T_{\mathrm{compute}}$，则该阶段通信成为瓶颈。

### 6.1 前向临界点

令 $T_{\mathrm{comm}}\ge T_{\mathrm{compute}}^{\mathrm{fwd}}$：

$$
\frac{4(N_{\mathrm{TP}}-1)\,BLD}{N_{\mathrm{TP}}\,W}
\ge
\frac{6\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,C}.
$$

两边同乘 $N_{\mathrm{TP}}$，约掉 $BLD$：

$$
\frac{4(N_{\mathrm{TP}}-1)}{W}
\ge
\frac{6\,D_{\mathrm{FF}}}{C}
\qquad\Longrightarrow\qquad
N_{\mathrm{TP}}-1
\ge
\frac{3}{2}\cdot D_{\mathrm{FF}}\cdot\frac{W}{C}.
$$

故前向在

$$
N_{\mathrm{TP}}
\ge
1 + \frac{3}{2}\,D_{\mathrm{FF}}\,\frac{W}{C}
$$

时通信瓶颈。

### 6.2 反向临界点

反向计算量是前向的两倍，通信量不变：

$$
\frac{4(N_{\mathrm{TP}}-1)\,BLD}{N_{\mathrm{TP}}\,W}
\ge
\frac{12\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}\,C}
\qquad\Longrightarrow\qquad
N_{\mathrm{TP}}-1
\ge
3\,D_{\mathrm{FF}}\,\frac{W}{C}.
$$

故反向在

$$
N_{\mathrm{TP}}
\ge
1 + 3\,D_{\mathrm{FF}}\,\frac{W}{C}
$$

时通信瓶颈。

### 6.3 读这两个式子

| | 临界点 |
|--|--------|
| 前向 | $N_{\mathrm{TP}}\ge 1+\tfrac{3}{2} D_{\mathrm{FF}} W/C$ |
| 反向 | $N_{\mathrm{TP}}\ge 1+3\,D_{\mathrm{FF}} W/C$ |

要点：

1. **$B$、$L$ 与 $D$ 被约掉了。** 通信量 $\propto BLD$，计算量 $\propto BLD D_{\mathrm{FF}}$，比值只剩 $D_{\mathrm{FF}}$ 与机器的 $C/W$。
2. **前向比反向更早撞上通信墙。** 同一份 all-reduce，反向有两倍 matmul，更能「养得起」通信；前向算得少，同样的通信更容易成为瓶颈。
3. **和数据并行对照。** DP 临界点是 $N_{\mathrm{DP}}\ge 1+B\,W/C$，瓶颈由 batch 决定；TP 临界点由 $D_{\mathrm{FF}}$ 决定。DP 靠加大 batch 推迟瓶颈；TP 靠更宽的中间层（或更高的 $C/W$，例如 NVLink）推迟瓶颈。
4. **这也解释了工程习惯：TP 通常只做在节点内。** 节点内 $W$ 大（NVLink），$W/C$ 大，临界 $N_{\mathrm{TP}}$ 才上得去；跨节点 $W$ 掉一个数量级，很小的 $N_{\mathrm{TP}}$ 就会通信主导。

---

## 7. 用 XL + 实验室带宽感受一下数量级

取 $D_{\mathrm{FF}}=10240$（XL）。若跨卡仍是 PCIe、$C/W\approx 4000$（与 DP 报告同一档）：

$$
\frac{W}{C}=\frac{1}{4000},
\qquad
\frac{3}{2} D_{\mathrm{FF}}\frac{W}{C}
=
\frac{3}{2}\cdot 10240\cdot\frac{1}{4000}
\approx 3.84.
$$

于是前向大约在 $N_{\mathrm{TP}}\ge 1+3.84\approx 4.8$ 起通信主导——也就是说，在 PCIe 上把 TP 开到 4 已经很紧，开到 8 前向几乎必通信瓶颈。

若换成节点内高带宽互联，把 $W$ 提高约 $10\times$，临界点大约也抬高约 $10\times$，这时 $N_{\mathrm{TP}}=8$ 才合理。这就是「TP 放节点内、DP/FSDP 跨节点」的定量版本。

---

## 8. 交作业时可以这样收束

**(a) 反向方程**见 §3.3：本地算出 $dW_1^{(i)},dW_2^{(i)},dW_3^{(i)}$ 与 $dx^{(i)}_{\mathrm{local}}$，再对 $\{dx^{(i)}_{\mathrm{local}}\}$ 做 all-reduce 得到 $dx$。

**(b) FLOPs（每设备）**

- 前向：$\dfrac{6\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$  
  （三个分片 matmul，每个 $2BLD D_{\mathrm{FF}}/N_{\mathrm{TP}}$。）
- 反向：$\dfrac{12\,B\,L\,D\,D_{\mathrm{FF}}}{N_{\mathrm{TP}}}$  
  （六个分片 matmul，每个同上。）

**(c) 通信时间**

- 前向与反向均为
  $$
  \frac{4(N_{\mathrm{TP}}-1)\,BLD}{N_{\mathrm{TP}}\,W}
  $$
  （各一次对 $(B\!\cdot\!L,\, D)$ FP16 张量的环形 all-reduce。）

**(d) 通信瓶颈临界点**

- 反向：$N_{\mathrm{TP}}\ge 1+3\,D_{\mathrm{FF}}\,W/C$
- 前向：$N_{\mathrm{TP}}\ge 1+\tfrac{3}{2}\,D_{\mathrm{FF}}\,W/C$

（令 $T_{\mathrm{comm}}\ge T_{\mathrm{compute}}$，约掉 $B,L,D$ 后解出。）
