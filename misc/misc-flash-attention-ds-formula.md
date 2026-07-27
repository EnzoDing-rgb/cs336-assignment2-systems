# Softmax 反向与 $D$ 恒等式（从定义推到数字验证）

本文分两步讲透两个核心问题，全程从定义出发，最后用数字例子验证。

**公式写法（方便 VS Code / Cursor Preview）：** 行内用 `$...$`，独立公式用 `$$...$$`（前后各空一行）。不用 `align` 环境。

---

## 一、先讲透 Softmax 反向公式：$dS_{ij} = P_{ij}(dP_{ij} - D_i)$

我们先只看**一行**注意力分数（Softmax 是逐行独立计算的，行与行之间互不影响），把行下标 $i$ 暂时去掉，只看一行内的第 $j$ 个位置。

### 1. 先明确正向规则

一行有 $n$ 个原始分数：

$$
S = [s_1,\, s_2,\, \ldots,\, s_n]
$$

Softmax 做两件事：

1. 每个位置取指数：$\exp(s_j)$
2. 归一化：每个位置除以所有指数的总和

分母记为：

$$
Z = \sum_{k=1}^{n} \exp(s_k)
$$

最终得到概率 $P = [p_1,\, p_2,\, \ldots,\, p_n]$，满足 $\sum_j p_j = 1$，公式为：

$$
p_j = \frac{\exp(s_j)}{Z}
$$

### 2. 反向要解决的问题

上游传回来了每个 $p_j$ 的梯度 $dp_j$（损失对 $p_j$ 的导数），我们要算出每个 $s_j$ 的梯度 $ds_j$（损失对原始分数 $s_j$ 的导数）。

这里有个关键耦合：**改动任意一个 $s_j$，会影响所有位置的 $p_k$**——因为分母 $Z$ 包含所有 $s$，一个位置变了，所有位置的归一化分母都会变，不是只影响自己。

### 3. 一步步求偏导

先算「$p_k$ 对 $s_j$ 的偏导数」，分两种情况。用商的导数法则：

$$
\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}
$$

#### 情况 1：$k = j$（自己对自己求导）

- 分子 $u = \exp(s_j)$，对 $s_j$ 求导：$u' = \exp(s_j)$
- 分母 $v = Z$，对 $s_j$ 求导：$v' = \exp(s_j)$

代入化简：

$$
\frac{\partial p_j}{\partial s_j}
= \frac{\exp(s_j)\cdot Z - \exp(s_j)\cdot \exp(s_j)}{Z^2}
= \frac{\exp(s_j)}{Z} - \left(\frac{\exp(s_j)}{Z}\right)^2
= p_j - p_j^2
= p_j(1 - p_j)
$$

#### 情况 2：$k \neq j$（其他位置对 $s_j$ 求导）

- 分子 $u = \exp(s_k)$，对 $s_j$ 求导：$u' = 0$（$s_k$ 和 $s_j$ 无关）
- 分母 $v = Z$，对 $s_j$ 求导：$v' = \exp(s_j)$

代入化简：

$$
\frac{\partial p_k}{\partial s_j}
= \frac{0\cdot Z - \exp(s_k)\cdot \exp(s_j)}{Z^2}
= -\frac{\exp(s_k)}{Z}\cdot\frac{\exp(s_j)}{Z}
= -p_k \cdot p_j
$$

### 4. 链式法则合并，得到最终公式

根据链式法则，$s_j$ 的梯度等于「所有 $p_k$ 的梯度 $\times$ 对应偏导」的总和：

$$
ds_j = \sum_{k=1}^{n} dp_k \cdot \frac{\partial p_k}{\partial s_j}
$$

把上面两种情况的偏导代入：

$$
ds_j = dp_j \cdot p_j(1 - p_j) + \sum_{k \neq j} dp_k \cdot (-p_k p_j)
$$

把公因子 $p_j$ 提出来：

$$
ds_j = p_j \left[ dp_j(1 - p_j) - \sum_{k \neq j} dp_k p_k \right]
$$

展开括号里的第一项：$dp_j - dp_j p_j$。

后面的求和是「除了 $j$ 之外所有位置的 $dp_k p_k$」，它等于「所有位置的和」减去「$j$ 位置自己」：

$$
\sum_{k \neq j} dp_k p_k = \sum_{k=1}^{n} dp_k p_k - dp_j p_j
$$

代入回括号里，中间项正好抵消：

$$
dp_j - dp_j p_j - \left( \sum_{k=1}^{n} dp_k p_k - dp_j p_j \right)
= dp_j - \sum_{k=1}^{n} dp_k p_k
$$

把这个求和项单独记为 $D$：

$$
D = \sum_{k=1}^{n} p_k \, dp_k
$$

最终得到简洁的反向公式：

$$
ds_j = p_j \cdot (dp_j - D)
$$

对应到矩阵写法就是：

$$
dS_{ij} = P_{ij}(dP_{ij} - D_i)
$$

其中 $i$ 是行号，每行独立计算自己的 $D_i$。

### 5. 数字例子验证

拿一行两个元素举例：

- $p = [0.25,\ 0.75]$
- $dp = [17,\ 39]$

第一步算 $D$：

$$
D = 0.25 \times 17 + 0.75 \times 39 = 4.25 + 29.25 = 33.5
$$

第二步算梯度：

$$
ds_1 = 0.25 \times (17 - 33.5) = 0.25 \times (-16.5) = -4.125
$$

$$
ds_2 = 0.75 \times (39 - 33.5) = 0.75 \times 5.5 = 4.125
$$

**直观理解：**

- 概率低的位置梯度为负，说明要降低它的原始分数
- 概率高的位置梯度为正，说明要提高它的原始分数
- 两个梯度相加为 $0$，符合「Softmax 概率和恒为 1；整体平移 $S$ 不改变 $P$」带来的约束感

更本质的检查：$\sum_j p_j \, ds_j = 0$。本例：

$$
0.25 \times (-4.125) + 0.75 \times 4.125 = 0
$$

---

## 二、再讲恒等式：$\mathrm{rowsum}(P \circ dP) = \mathrm{rowsum}(O \circ dO)$

也就是为什么 $D$ 既可以用 $P$ 算，也可以用 $O$ 算。

### 1. 先明确 $O$ 和 $P$ 的关系

注意力的输出 $O$，本质是用 $P$ 对 $V$ 做加权求和：

$$
O = \sum_{j=1}^{n} p_j \cdot v_j
$$

- $v_j$ 是第 $j$ 个 value 向量，维度为 $d$（比如 64）
- 写成矩阵形式：$O = P V$（这里把一行 $P$ 看成 $1 \times n$，$V$ 是 $n \times d$，$O$ 是 $1 \times d$）

### 2. 先推 $dP$ 和 $dO$ 的关系

反向时，我们已经有了上游传来的 $dO$（$1 \times d$），要算 $dP$（$1 \times n$）。

根据链式法则，第 $j$ 个权重的梯度 $dp_j$，等于「所有输出维度的梯度 $\times$ 对应偏导」的和：

$$
dp_j = \sum_{c=1}^{d} dO_c \cdot \frac{\partial O_c}{\partial p_j}
$$

而

$$
O_c = \sum_j p_j \, v_{jc}
$$

对 $p_j$ 求偏导正好就是 $v_{jc}$。所以：

$$
dp_j = \sum_{c=1}^{d} dO_c \cdot v_{jc}
$$

也就是：$dp_j$ 等于 $dO$ 和第 $j$ 个 $v_j$ 的点积。

写成矩阵形式：

$$
dP = dO \, V^{\mathsf{T}}
$$

形状匹配：$dO$ 是 $1 \times d$，$V^{\mathsf{T}}$ 是 $d \times n$，相乘结果是 $1 \times n$，和 $P$ 形状一致。转置 $V$ 是为了矩阵乘法维度匹配。

### 3. 代入 $D$ 的定义，证明两边相等

$D$ 的定义是：

$$
D = \sum_j p_j \, dp_j
$$

（也就是 $\mathrm{rowsum}(P \circ dP)$。）

把 $dp_j = \sum_c dO_c \, v_{jc}$ 代入：

$$
D = \sum_{j=1}^{n} p_j \cdot \sum_{c=1}^{d} dO_c \, v_{jc}
$$

交换两个求和的顺序（有限和，可交换）：

$$
D = \sum_{c=1}^{d} dO_c \cdot \sum_{j=1}^{n} p_j \, v_{jc}
$$

右边的 $\sum_j p_j v_{jc}$ 正好是输出 $O$ 的第 $c$ 维 $O_c$。代入后：

$$
D = \sum_{c=1}^{d} dO_c \cdot O_c
$$

右边就是 $\mathrm{rowsum}(O \circ dO)$（逐元素相乘再按行求和）。

至此：

$$
\sum_j p_j \, dp_j = \sum_c O_c \, dO_c
$$

### 4. 数字例子验证

沿用前面的数值，并取：

$$
V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

（2 个 value，每个 2 维；第 1 行是 $v_1=(1,2)$，第 2 行是 $v_2=(3,4)$。）

- $p = [0.25,\ 0.75]$
- $dO = [5,\ 6]$

**先算正向 $O$：**

$$
O_1 = 0.25 \times 1 + 0.75 \times 3 = 2.5
$$

$$
O_2 = 0.25 \times 2 + 0.75 \times 4 = 3.5
$$

$$
O = [2.5,\ 3.5]
$$

**算右边 $\mathrm{rowsum}(O \circ dO)$：**

$$
2.5 \times 5 + 3.5 \times 6 = 12.5 + 21 = 33.5
$$

左边之前已算过 $\mathrm{rowsum}(P \circ dP) = 33.5$，两边完全相等。

### 5. 直觉理解

这不是巧合，是线性混合的固有性质：

- $P$ 是权重，$V$ 是各 key 的 value，$O$ 是加权混合后的结果
- 「权重的梯度按权重加权和」和「输出的梯度与输出点积」，是同一个变化量的两种记账方式
- 类似：总支出变化既可以按「各品类贡献」加总，也可以直接看「总价变化」，结果必须一致

---

## 三、回到 FlashAttention：为什么要费这个劲？

核心目的只有一个：**省显存**。

1. $P$ 是 $N \times N$ 的大矩阵，序列越长占显存越多。FlashAttention 正向算完注意力权重后不把完整 $P$ 长期留在 HBM。
2. $O$ 是 $N \times d$ 的矩阵，$d$ 是头维度（通常 64/128），比 $N$ 小一到两个数量级，存下来便宜得多；再存一行一个标量的 $L$（logsumexp），用来反向重算 $P_{ij} = \exp(S_{ij} - L_i)$。
3. 反向算 $dS$ 必须用到 $D$。本来 $D = \mathrm{rowsum}(P \circ dP)$ 看起来离不开 $P$；现在用提前存好的 $O$ 加上游传来的 $dO$ 就能算出同一个 $D$，再临时重算 $P$，就能算出梯度。

全程不必把完整的 $N \times N$ 矩阵从正向一直存到反向——这是 FlashAttention 能用较小激活显存跑更长序列的核心技巧之一。
