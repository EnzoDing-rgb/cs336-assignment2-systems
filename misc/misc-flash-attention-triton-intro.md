# Flash Attention 学习前置：Triton 与加权求和范例

> 对应讲义 **§4.2.1 加权求和（Weighted Sum）** 范例：前向、反向内核与 `autograd` 衔接。  
> 目的：在实现 **Flash Attention 第二版（FlashAttention-2）** 之前，先弄懂 **Triton 如何与 PyTorch 自动微分（autograd）协作**。

---

## 0. 读完本文你应该能回答什么

1. 为什么 `torch.compile` 之后还要手写 Triton 实现注意力？
2. 分块结果和裸算一样，**好处**是什么（访存、并行）？
3. Triton 内核与 PyTorch 算子的根本差别是什么（指针、步长、分块）？
4. 孤立内核如何接到 **自动微分（autograd）**，让 `loss.backward()` 能反传？
5. 如何用 **块指针（block pointer）** 在全局内存里搬一块 tile？
6. 加权求和反向里，为什么 `grad_weight` 要 **分块累加再求和**，而 `grad_x` 却 **直接写进最终缓冲区**？

---

## 1. 从编译注意力到 Flash Attention：为什么换一条路

### 1.1 上一节作业交付什么

讲义在进入 Triton 之前，先收束 **即时编译（just-in-time compile，`torch.compile`）** 的两项对比：

| 部分 | 对比对象 | 交付物 |
|------|----------|--------|
| 注意力算子 | 编译版 vs 上一题未编译版 | 前向、反向时间对照表 |
| 整网 Transformer | 编译版 vs 端到端基准里的未编译版 | 前向、前向+反向+优化器步对照表 |

这些实验说明：`torch.compile` 能通过算子融合减少调度开销和部分高带宽内存（high bandwidth memory，HBM）往返，**但长序列时仍不够**。

### 1.2 为什么还要 Flash Attention

讲义原话的核心意思：

- 序列长度变大时，即便用了 `torch.compile`，当前实现仍在 **访存模式** 上很差；
- 因此要用 **Triton** 手写 **Flash Attention 第二版**，从而 **精确控制何时从哪读内存、何时算什么**。

一句话：**编译器帮你融算子，但融不掉「必须把整块注意力矩阵写在 HBM 上」这件事；Flash Attention 从算法层面改掉这一点。**

---

## 2. 热身范例：加权求和（Weighted Sum）

讲义 **§4.2.1** 不用一上来就写注意力，而是用一个更小的算子教你 Triton 与 PyTorch 的衔接。读懂这一节，后面 Flash Attention 内核只是 **分块更复杂、数学更多步**。

### 2.1 这个算子在算什么

给定矩阵 `X`（形状 `[..., D]`，最后一维长度为 `D`）和 **`weight`（手算时画成 1×4 横排，代码里形状 `[D]`）**：

- 先把 `X` 的每一行与 `weight` **逐元素相乘**；
- 再沿最后一维 **求和**；
- 得到每个「行位置」一个标量。

PyTorch 前向：

```python
def weighted_sum(x, weight):
    # x: [..., D], weight: [D]
    return (weight * x).sum(axis=-1)
```

若把 `x` 展平成二维矩阵 `(行数, D)`，这就是 **矩阵 `X` 与向量 `weight` 的矩阵-向量积**：每一行与 `weight` 做内积。

**推广（§5 手算用这个）：** 若 `weight` 是 **$K$ 行 $D$ 列** 的矩阵，则每一行 `X` 与 `weight` 的 **每一行** 各做一次内积，得到 **$K$ 个输出**。`§3.3` 的前向是 **$K=1$** 的特例（`weight` 只有一行）；讲义里的实际代码也是 $K=1$。§5 用 **$K=2$** 举例，因为一眼能看出「多路输出」时反向公式怎么变，以及分块时 **`grad_x` 与 `grad_weight` 为何不对称**。

**输入：** 矩阵 `x ∈ ℝ^{n×h}`，权重 `w ∈ ℝ^h`（讲义用 `n` 表示行数，`h` 表示特征维 `D`）。  
**输出：** 向量 `y ∈ ℝ^n`，其中 `y_i = Σ_j w_j · x_{ij}`。

---

## 3. Triton 前向：和 PyTorch 差在哪

### 3.1 执行模型（先建立直觉）

- **程序实例（program instance）**：GPU 上 **并行** 跑的多份「同一份内核程序」。每份实例用 **`tl.program_id(轴号)`** 知道自己排第几——**不是** SWIG、也不是别的框架 ID，就是 Triton 的 program id（类似 CUDA 的 `blockIdx`，但 Triton 里统一叫 program）。
- Triton 内核 **不直接收 PyTorch 张量**，而是收：
  - 指向张量 **首元素** 的指针；
  - 每个维度的 **步长（stride）**：沿某一维走一格，在内存里要跳过多少元素。
- 与 PyTorch 相比，多出来的事主要是：**指针运算、显式加载（load）、显式写回（store）**。

讲义采用较新的 **块指针（block pointer）** 抽象（`tl.make_block_ptr`），用声明式方式描述「从哪块内存读/写多大一块」，减轻手写指针算术。

### 3.2 分块策略（对应讲义图 2）

矩阵 `X` 按 **行** 切成若干 tile，每个 tile 高度为 `ROWS_TILE_SIZE`；沿特征维 `D` 再切成宽度为 `D_TILE_SIZE` 的小条。

- 第 `i` 个程序实例负责 **第 `i` 块行 tile**；代码里 `i = tl.program_id(0)`，手算里叫 **实例 0、实例 1**（从 0 开始编号）；
- 沿 `D` 方向用循环，每次处理 `D_TILE_SIZE` 列；
- 块指针用 `.advance((行方向增量, 列方向增量))` 移到下一条带。

下面用一个 **4×4 矩阵、2×2 分块** 的完整例子，把每一块指针在读什么、写什么钉死。

### 3.3 完整手算例子：4×4 矩阵，按 2×2 分块

#### 3.3.1 输入数据（每个位置数字不同，便于对照）

设 `NUM_ROWS = 4`，`D = 4`，`ROWS_TILE_SIZE = 2`，`D_TILE_SIZE = 2`。  
矩阵 `x`（行优先存储，位置 `(行, 列)` 的值为）：

```
         列0   列1   列2   列3
行0       1     2     3     4
行1       5     6     7     8
行2       9    10    11    12
行3      13    14    15    16
```

权重 `weight = [10, 20, 30, 40]`：**手算时画成 1×4 横排**（一行四列）；代码里 PyTorch 用一维 `(4,)` 存，是同一组数。

加权求和：第 `i` 行输出  
`y[i] = x[i,0]×10 + x[i,1]×20 + x[i,2]×30 + x[i,3]×40`。

**PyTorch 手算正确答案：**

| 行 | 计算 | `y[i]` |
|----|------|--------|
| 0 | 1×10 + 2×20 + 3×30 + 4×40 | **300** |
| 1 | 5×10 + 6×20 + 7×30 + 8×40 | **700** |
| 2 | 9×10 + 10×20 + 11×30 + 12×40 | **1100** |
| 3 | 13×10 + 14×20 + 15×30 + 16×40 | **1500** |

输出 `y`：**4×1 竖列**（也可记作长度 4 的一维向量 `[300, 700, 1100, 1500]`）。  
`X` 的每一行只产出一个标量，所以 `y` 有 4 个数，第 $i$ 个对应 `X` 的第 $i$ 行：

```text
y (4×1)          计算来源
  300    ←  X 第 0 行 × weight 横排
  700    ←  X 第 1 行 × weight 横排
 1100    ←  X 第 2 行 × weight 横排
 1500    ←  X 第 3 行 × weight 横排
```

#### 3.3.2 谁干什么：两个程序实例

`launch grid` 大小 = `ceil(4 / 2) = 2`：

| `row_tile_idx` | 负责的行 | 要写回的 `y` 下标 |
|----------------|----------|---------------------|
| 0 | 行 0、行 1 | `y[0]`、`y[1]` |
| 1 | 行 2、行 3 | `y[2]`、`y[3]` |

沿列方向还要循环两轮（`D / D_TILE_SIZE = 4/2 = 2`）：  
第 0 轮读列 0–1，第 1 轮读列 2–3；每轮把 **部分和** 累加进寄存器 `output`，两轮结束后才是完整内积。

---

#### 3.3.3 程序实例 0（`row_tile_idx = 0`）逐步对照伪代码

**① 三个块指针指向哪里**

```text
x_block_ptr:
  shape=(4,4), offsets=(0,0), block_shape=(2,2)
  → 盯住 x 的左上角 2×2 块：
       1   2
       5   6

weight_block_ptr:
  shape=(4,), offsets=(0,), block_shape=(2,)
  → 盯住 weight 的前半段：[10, 20]

output_block_ptr:
  shape=(4,), offsets=(0,), block_shape=(2,)
  → 盯住输出向量 y 的下标 0、1 这两个位置（即将写入 y[0], y[1]）
```

**② 寄存器 `output = tl.zeros((2,))` → `[0, 0]`**

**③ 循环第 0 轮（`i = 0`，列 0–1）**

```text
row = load(x_block_ptr)     →  [[1, 2], [5, 6]]
weight = load(weight_ptr)   →  [10, 20]

row * weight[None, :]       →  [[1×10, 2×20], [5×10, 6×20]]
                            =  [[10, 40], [50, 120]]

tl.sum(..., axis=1)         →  [50, 170]    # 行0: 10+40；行1: 50+120

output +=                   →  [50, 170]

advance: x 列指针 +2 → 下轮读列 2–3；weight +2 → 下轮读 [30, 40]
```

**④ 循环第 1 轮（`i = 1`，列 2–3）**

```text
row = load(x_block_ptr)     →  [[3, 4], [7, 8]]
weight = load(weight_ptr)   →  [30, 40]

tl.sum(...)                 →  [3×30+4×40, 7×30+8×40] = [250, 530]

output +=                   →  [50+250, 170+530] = [300, 700]  ✓ 与手算一致
```

**⑤ `tl.store(output_block_ptr, [300, 700])`**

写入 `y[0]=300`，`y[1]=700`。

---

#### 3.3.4 程序实例 1（`row_tile_idx = 1`）简要

`offsets` 行方向 = `1 × 2 = 2`，故：

```text
x 盯住行 2–3：
       9  10
      13  14        （第 0 轮列 0–1）

output_block_ptr offsets=(2,), block_shape=(2,)
  → 写入 y[2], y[3]
```

两轮循环后 `output = [1100, 1500]`，store 到 `y[2]`、`y[3]`。

---

#### 3.3.5 为什么 `output_block_ptr` 的 `block_shape` 是 `(ROWS_TILE_SIZE,)`？

这是最容易困惑的一点，拆开看：

| 张量 | 逻辑形状 | 本程序实例一次处理什么 | `block_shape` |
|------|----------|------------------------|---------------|
| `x` | `(NUM_ROWS, D)` = `(4, 4)` 二维 | `2` 行 × `2` 列的子矩阵 | `(2, 2)` |
| `weight` | `(D,)` = `(4,)` 一维 | 当前列条带上 `2` 个权重 | `(2,)` |
| **`y`（输出）** | **`(NUM_ROWS,)` = `(4,)` 一维** | **本实例负责的 `2` 个行的标量结果** | **`(2,)`** |

输出 `y` **没有列维**：每一行算完后只有一个数。  
程序实例 0 算的是行 0、行 1 的两个标量，所以一次 store 写 **2 个元素** → `block_shape=(ROWS_TILE_SIZE,)=(2,)`。  
`shape=(NUM_ROWS,)` 表示整个输出向量总长 4；`offsets=(row_tile_idx * 2,)` 表示从 `y` 的第几条开始写。

**一句话：`ROWS_TILE_SIZE` 在输出上表示「本实例一次写回几个行的标量」，不是矩阵的行高列宽。**

---

### 3.4 前向内核（讲义代码；上节例子即其执行过程）

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)  # 例子里：0 或 1

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),  # 例：实例0→行0，实例1→行2
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),                           # 整个 y 长度 4
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),    # 例：实例0→写 y[0:2]
        block_shape=(ROWS_TILE_SIZE,),               # 一次写 2 个标量
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)  # 例：[0,0] 再累加

    for i in range(tl.cdiv(D, D_TILE_SIZE)):        # 例：循环 2 轮（列 0–1，列 2–3）
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))

    tl.store(output_block_ptr, output, boundary_check=(0,))  # 例：写入 [300,700] 或 [1100,1500]
```

**逐段在干什么：**

| 代码段 | 含义 |
|--------|------|
| `make_block_ptr` 六个参数 | 首地址、张量总形状、步长、本块起始偏移、本块形状、内存维顺序（主序到次序，用于优化） |
| `boundary_check` + `padding_option="zero"` | 行数或 `D` 不能整除 tile 大小时，越界处填零，避免读垃圾 |
| 循环内 `load` | 读当前 `D` 条带上的 `x` 与 `weight` |
| `tl.sum(row * weight[None,:], axis=1)` | 对每行做 `Σ_j x_ij * w_j`，累加到 `output` |
| `advance` | 块指针沿 `D` 移到下一条带 |
| `store` | 把本 tile 各行标量写回输出向量 |

### 3.5 分块和裸算结果一样，好处在哪？

§3.3 已经验证：**分块只是把同一套加法换了一种执行顺序**，`y` 仍是 `[300, 700, 1100, 1500]`。  
那为什么要费劲写 Triton、搞块指针？

**第一性原理：GPU 有两层「仓库」。**

| 仓库 | 典型名称 | 容量 | 速度 |
|------|----------|------|------|
| 片外 | 高带宽内存（HBM） | 很大（几十 GiB） | 相对慢 |
| 片上 | 共享内存 / 寄存器 | 很小（每程序实例 KB 级） | 很快 |

**裸算（PyTorch 一行 `(weight * x).sum(-1)`）在干什么：**

1. 调度器启动一个（或几个）通用内核；
2. 往往要把整块 `x`、`weight` 从 HBM 读进来、算完、再把整块中间结果写回 HBM；
3. 对加权求和这种简单算子，差别不大——**所以本节只是教学范例，加速不明显**。

**分块（Triton）在干什么：**

1. **并行**：程序实例 0 算行 0–1，程序实例 1 算行 2–3，**同时跑**（例子里的 2 路并行；真实训练里可能是成百上千块）；
2. **少搬大数据**：每次只 `load` 一个 2×2 小 tile 到片上，在寄存器里乘加、累加，**不必为每一步都申请一块完整大小的中间张量写到 HBM**；
3. **算子融合的前奏**：循环里的乘、加、累加可以在 **同一次内核** 里完成，不必拆成「乘法内核 → 求和内核」两次往返 HBM。

用 §3.3 的例子说：

```text
裸算思路：  一次性读入整个 4×4 的 x 和整个 weight → 算 → 写出 y
分块思路：  读 2×2 → 在寄存器累加 → 再读下一块 2×2 → 再累加 → 最后只写 2 个标量
            （两个程序实例各干各的行，互不等待）
```

**本节加权求分块的主要收益是「学会套路」，不是刷榜速度。**  
后面 Flash Attention 用 **同一套分块 + 在片上累加** 去避免写出 `(序列长度 × 序列长度)` 的注意力矩阵——那时 HBM 往返从「可忽略」变成「致命瓶颈」。  
先在这里把 **分块 = 正确性不变 + 控制访存 + 并行** 记住，后面只是矩阵更大、公式更多。

---

## 4. 从孤立内核到可训练：为什么要接 PyTorch？

到这里为止，我们有一个 **`weighted_sum_fwd` 内核**：给它指针，它在 GPU 上算出正确的 `y`。  
但训练不是「算一次 `y`」就结束。

### 4.1 训练时 PyTorch 在跟踪什么

一次训练步大致是：

```text
y = weighted_sum(x, weight)   # 前向
loss = …(y, …)                # 损失
loss.backward()               # 反向：需要知道每个输入的梯度
optimizer.step()              # 用梯度更新 weight 等参数
```

`loss.backward()` 时，PyTorch 会沿 **计算图（computation graph）** 从 `loss` 往回走：  
每个算子必须回答——**「给定上游传来的梯度，我对自己每个输入的梯度是多少？」**

对内置算子（例如 `torch.matmul`），PyTorch 已经内置了答案。  
对我们的 `weighted_sum_fwd`：**PyTorch 一无所知**——它只是一段你写的 GPU 代码，图里没有一个「节点」对应它。

### 4.2 缺的那一环：谁告诉 autograd 怎么反传？

需要人为补一张「说明书」：

| 阶段 | 谁调用 | 输入 | 输出 | 还要做什么 |
|------|--------|------|------|------------|
| 前向 | `forward` | `x`, `weight` | `y` | 把反向要用到的 `x`, `weight` **存起来** |
| 反向 | `backward` | 上游的 `grad_output`（即 ∂loss/∂y） | `grad_x`, `grad_weight` | 读回存下的张量，启动反向内核 |

**`torch.autograd.Function`** 就是这份说明书的固定格式：  
你实现 `forward` 和 `backward` 两个静态方法，PyTorch 在前向时登记节点，在 `backward()` 时自动调你的 `backward`。

逻辑链（无断点）：

```text
用户调用 WeightedSumFunc.apply(x, weight)
    → PyTorch 调你的 forward
        → 整理张量形状、分配输出缓冲区
        → 启动 weighted_sum_fwd 内核（§3 的分块算法）
        → save_for_backward(x, weight)
        → 返回 y（y 上挂 grad_fn，标记「反向请找我」）
    … 后面 loss.backward() …
    → PyTorch 调你的 backward(grad_output)
        → 启动 weighted_sum_backward 内核（§6）
        → 返回 (grad_x, grad_weight)
```

**不是「打包进 PyTorch」这种模糊说法，而是：给自定义 GPU 内核补上 autograd 认识的前向/反向接口。**

### 4.3 前向 `forward`：逐行在干什么（对照 §3.3）

下面用 **4×4 例子** 的语义读真实代码（真实训练里 `ROWS_TILE_SIZE=16`，块更大，道理相同）。

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # ── 1. 弄清形状 ──
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")
```

`x` 可能有批量维，例如 `(2, 3, 4)` 表示 2×3=6 行、每行 4 列。  
内核只认识二维 `(行数, D)`，所以展平成 `(6, 4)`。§3.3 的 4×4 就是 `行数=4, D=4` 的特例。

```python
        ctx.save_for_backward(x, weight)
```

反向要用 §5.2 的公式（$K=1$ 时 `grad_x[i,j] = grad_output[i] * weight[j]` 等），  
必须把 **前向时的** `x` 和 `weight` 存进 `ctx`（或像 Flash Attention 那样只存检查点——加权求和这里直接全存）。

```python
        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device)
        n_rows = y.numel()
```

在 GPU 上 **先开好输出缓冲区** `y`（长度 = 行数；4×4 例子里是 4）。  
`empty` 不填零，因为内核会 **完整覆盖** 每个位置——§3.3 最终写入 `[300,700,1100,1500]`。

```python
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])
```

**启动内核：**

- `[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)]`：方括号里是 **launch grid**。  
  §3.3 里 `n_rows=4, ROWS_TILE_SIZE=2` → grid 长度 2（两个程序实例）。  
  真实代码用 `ROWS_TILE_SIZE=16`，行数很大时 grid 也会很大，**每个实例仍只负责自己的行块**。
- 传入 `x, weight, y`：PyTorch 张量在底层就是指针 + 步长，对应内核里的 `x_ptr` 与 `stride`。
- 内核跑完后，`y` 里已是最终结果；`view` 把 `(6,)` 还原成 `(2, 3)` 等原始前缀形状。

**对照 §3.3 的完整数据流：**

```text
PyTorch 张量 x (4×4), weight (4,)
    → forward 展平、分配 y (长度 4)
    → launch grid = 2
    → 实例 0 写 y[0], y[1]；实例 1 写 y[2], y[3]
    → 返回 y，grad_fn 指向 WeightedSumFuncBackward
```

### 4.4 前向里还要留心的检查

- `weight` 一维且长度等于 `D`；
- `x`、`weight` 在 CUDA 设备上（内核是 GPU 代码）；
- `x` **连续存储**：步长 `(stride(0), stride(1))` 假设行优先、无空洞；否则块指针会指错位置。

---

## 5. 反向传播：公式、完整手算、分块对比

§4 解决了前向：`y = f(x, weight)` 由 `weighted_sum_fwd` 算出，并 `save_for_backward(x, weight)`。  
`loss.backward()` 时，PyTorch 传入 **损失对 `y` 的梯度** `grad_output`，要求我们算出 **损失对 `x` 和 `weight` 的梯度**。

本节分 **两件事** 讲清楚：

1. **反向公式为什么长这样**（先 **不分块**，把每个数算对）；
2. **分块实现时为何不对称**——有 `partial_grad_weight`，却没有 `partial_grad_x`（用 **同一组数字** 走两遍：先整体算，再按代码分块算）。

手算统一用下面这组数（与 §3.3 的 `X` 相同，但 `weight` 扩成 2 行、`grad_output` 扩成 2 列，泛化性更好）：

```text
X (4×4)                              weight (2×4)
         列0   列1   列2   列3              列0  列1  列2  列3
行0       1     2     3     4         行0    10   20   30   40    ← 输出通道 k=0
行1       5     6     7     8         行1    50   60   70   80    ← 输出通道 k=1
行2       9    10    11    12
行3      13    14    15    16

grad_output (4×2，与 y 同形)              记号
         列0   列1                    ∂L/∂y[i,k]
行0     100   500
行1     200   600
行2     300   700
行3     400   800

# 展平（行优先）即：100, 500, 200, 600, 300, 700, 400, 800
```

分块参数与 §3.3、§6 代码一致：`ROWS_TILE_SIZE = 2`，`D_TILE_SIZE = 2` → 行方向 2 个程序实例，列方向循环 2 轮。

---

### 5.1 反向在要什么

| 传入 / 要算 | 形状（本例） | 含义 |
|-------------|--------------|------|
| **`grad_output`** | 4×2 | $\partial \mathcal{L}/\partial y_{ik}$：$y$ 在第 $i$ 行、第 $k$ 个输出通道增加 1 时，$\mathcal{L}$ 增加多少 |
| **`grad_x`** | 4×4 | $\partial \mathcal{L}/\partial x_{ij}$ |
| **`grad_weight`** | 2×4 | $\partial \mathcal{L}/\partial w_{kj}$（$k$ = `weight` 行号，$j$ = 列号） |

**下标约定（全文统一）：**

- $i \in \{0,1,2,3\}$：`X` 的 **行号**，也是 `y`、`grad_output` 的行号；
- $j \in \{0,1,2,3\}$：`X` 的 **列号**，也是 `weight` 的列号；
- $k \in \{0,1\}$：`weight` 的 **行号**（第几个输出通道），也是 `y`、`grad_output` 的列号。

---

### 5.2 前向与反向公式（带求和上下限）

#### 前向

对每一个行号 $i \in \{0,1,2,3\}$、每一个输出通道 $k \in \{0,1\}$：

$$
y_{ik} = \sum_{j=0}^{3} w_{kj} \cdot x_{ij}
$$

读法：固定 `X` 的第 $i$ 行，与 `weight` 的第 $k$ 行做内积。  
**$K=1$ 时** `weight` 只有一行，$y$ 退化成 §3.3 的 4×1 竖列。

**例子（$i=0,\,k=0$）：**

```text
y[0,0] = 1·10 + 2·20 + 3·30 + 4·40 = 300
y[0,1] = 1·50 + 2·60 + 3·70 + 4·80 = 700   （weight 第 1 行）
```

完整 $y$（4×2）：

```text
         k=0    k=1
行0      300    700
行1      700   1740
行2     1100   2780
行3     1500   3820
```

#### 反向

**`grad_x`：** 固定 $(i,j)$，`x[i,j]` 出现在 **本行** 的 **每一个** 输出通道 $y_{ik}$ 里（$k=0,1$ 各一项），要把两路贡献 **相加**：

$$
\frac{\partial \mathcal{L}}{\partial x_{ij}}
= \sum_{k=0}^{1} w_{kj} \cdot \frac{\partial \mathcal{L}}{\partial y_{ik}}
$$

**`grad_weight`：** 固定 $(k,j)$，`weight[k,j]` 出现在 **每一行** 的 $y_{ik}$ 里（$i=0,1,2,3$ 各一项），要把四行贡献 **相加**：

$$
\frac{\partial \mathcal{L}}{\partial w_{kj}}
= \sum_{i=0}^{3} x_{ij} \cdot \frac{\partial \mathcal{L}}{\partial y_{ik}}
$$

代码对应（$K=2$ 时）：

```text
grad_x[i,j]       = sum over k=0..1 of  grad_output[i,k] * weight[k,j]
grad_weight[k,j]  = sum over i=0..3 of  grad_output[i,k] * x[i,j]
```

**$K=1$ 特例（讲义代码）：** `weight` 一行、`grad_output` 一列，两个求和里各只剩一项 → `grad_x[i,j] = grad_output[i] * weight[j]`，`grad_weight[j] = sum_i grad_output[i] * x[i,j]`。

---

### 5.3 第一件事：不分块，完整手算

#### 5.3.1 算 `grad_x`——从 `x[0,0]` 推起

`x[0,0]=1` 只出现在 **第 0 行** 的两个输出里：

```text
y[0,0] = 10·x[0,0] + …        →  ∂y[0,0]/∂x[0,0] = 10
y[0,1] = 50·x[0,0] + …        →  ∂y[0,1]/∂x[0,0] = 50
```

两行 $i=1,2,3$ 的公式里 **只含** 各自行的 $x_{i'j}$（$i'\neq 0$），所以 $x[0,0]$ 只通过 $k=0$ 和 $k=1$ 两条路影响 $\mathcal{L}$：

```text
∂L/∂x[0,0] = grad_output[0,0]·10 + grad_output[0,1]·50
            = 10·100 + 50·500
            = 1000 + 25000
            = 26000
```

**同行其余列**（$i=0$ 固定，对 $k$ 求和）：

```text
grad_x[0,1] = 20·100 + 60·500  = 32000
grad_x[0,2] = 30·100 + 70·500  = 38000
grad_x[0,3] = 40·100 + 80·500  = 44000
```

**第 1 行**（$i=1$，用 `grad_output` 第 1 行 `[200, 600]`）：

```text
grad_x[1,0] = 10·200 + 50·600  = 2000 + 30000  = 32000
grad_x[1,1] = 20·200 + 60·600  = 4000 + 36000  = 40000
grad_x[1,2] = 30·200 + 70·600  = 6000 + 42000  = 48000
grad_x[1,3] = 40·200 + 80·600  = 8000 + 48000  = 56000
```

**第 2 行**（$i=2$，用 `grad_output` 第 2 行 `[300, 700]`）：

```text
grad_x[2,0] = 10·300 + 50·700  = 3000 + 35000  = 38000
grad_x[2,1] = 20·300 + 60·700  = 6000 + 42000  = 48000
grad_x[2,2] = 30·300 + 70·700  = 9000 + 49000  = 58000
grad_x[2,3] = 40·300 + 80·700  = 12000 + 56000 = 68000
```

**第 3 行**（$i=3$，用 `grad_output` 第 3 行 `[400, 800]`）：

```text
grad_x[3,0] = 10·400 + 50·800  = 4000 + 40000  = 44000
grad_x[3,1] = 20·400 + 60·800  = 8000 + 48000  = 56000
grad_x[3,2] = 30·400 + 70·800  = 12000 + 56000 = 68000
grad_x[3,3] = 40·400 + 80·800  = 16000 + 64000 = 80000
```

**完整 `grad_x` 验算清单（16 格，逐格代入）：**

```text
grad_x[0,0] = 10·100 + 50·500 = 26000
grad_x[0,1] = 20·100 + 60·500 = 32000
grad_x[0,2] = 30·100 + 70·500 = 38000
grad_x[0,3] = 40·100 + 80·500 = 44000
grad_x[1,0] = 10·200 + 50·600 = 32000
grad_x[1,1] = 20·200 + 60·600 = 40000
grad_x[1,2] = 30·200 + 70·600 = 48000
grad_x[1,3] = 40·200 + 80·600 = 56000
grad_x[2,0] = 10·300 + 50·700 = 38000
grad_x[2,1] = 20·300 + 60·700 = 48000
grad_x[2,2] = 30·300 + 70·700 = 58000
grad_x[2,3] = 40·300 + 80·700 = 68000
grad_x[3,0] = 10·400 + 50·800 = 44000
grad_x[3,1] = 20·400 + 60·800 = 56000
grad_x[3,2] = 30·400 + 70·800 = 68000
grad_x[3,3] = 40·400 + 80·800 = 80000
```

汇总成表：

```text
              列0     列1     列2     列3
行0         26000   32000   38000   44000
行1         32000   40000   48000   56000
行2         38000   48000   58000   68000
行3         44000   56000   68000   80000
```

**规律：** 固定行 $i$、列 $j$，对 **输出通道 $k$** 求和：`grad_output[i,k] × weight[k,j]`。  
（$K=1$ 时求和只有一项，就是之前的「只做乘法」。）

#### 5.3.2 算 `grad_weight`——从 `weight[0,0]` 推起

`weight[0,0]=10` 在 **四行** 的 $y_{i,0}$ 里各出现一次：

```text
∂L/∂weight[0,0]
  = grad_output[0,0]·x[0,0] + grad_output[1,0]·x[1,0]
  + grad_output[2,0]·x[2,0] + grad_output[3,0]·x[3,0]
  = 1·100 + 5·200 + 9·300 + 13·400
  = 100 + 1000 + 2700 + 5200
  = 9000
```

**`weight` 第 0 行其余列：**

```text
grad_weight[0,1] = 2·100 + 6·200 + 10·300 + 14·400
                   = 200 + 1200 + 3000 + 5600
                   = 10000

grad_weight[0,2] = 3·100 + 7·200 + 11·300 + 15·400
                   = 300 + 1400 + 3300 + 6000
                   = 11000

grad_weight[0,3] = 4·100 + 8·200 + 12·300 + 16·400
                   = 400 + 1600 + 3600 + 6400
                   = 12000
```

**`weight` 第 1 行**（$k=1$，用 `grad_output` 第 1 列 500, 600, 700, 800）：

```text
grad_weight[1,0] = 1·500 + 5·600 + 9·700 + 13·800
                 = 500 + 3000 + 6300 + 10400
                 = 20200

grad_weight[1,1] = 2·500 + 6·600 + 10·700 + 14·800
                 = 1000 + 3600 + 7000 + 11200
                 = 22800

grad_weight[1,2] = 3·500 + 7·600 + 11·700 + 15·800
                 = 1500 + 4200 + 7700 + 12000
                 = 25400

grad_weight[1,3] = 4·500 + 8·600 + 12·700 + 16·800
                 = 2000 + 4800 + 8400 + 12800
                 = 28000
```

**完整 `grad_weight` 验算清单（8 格，逐格代入）：**

```text
grad_weight[0,0] = 1·100 + 5·200 + 9·300 + 13·400 = 9000
grad_weight[0,1] = 2·100 + 6·200 + 10·300 + 14·400 = 10000
grad_weight[0,2] = 3·100 + 7·200 + 11·300 + 15·400 = 11000
grad_weight[0,3] = 4·100 + 8·200 + 12·300 + 16·400 = 12000
grad_weight[1,0] = 1·500 + 5·600 + 9·700 + 13·800 = 20200
grad_weight[1,1] = 2·500 + 6·600 + 10·700 + 14·800 = 22800
grad_weight[1,2] = 3·500 + 7·600 + 11·700 + 15·800 = 25400
grad_weight[1,3] = 4·500 + 8·600 + 12·700 + 16·800 = 28000
```

汇总成表：

```text
              列0     列1     列2     列3
k=0          9000   10000   11000   12000
k=1         20200   22800   25400   28000
```

**规律：** 固定 `weight` 的 $(k,j)$，对 **行号 $i$** 求和：`grad_output[i,k] × x[i,j]`。

#### 5.3.3 对照：谁沿哪个方向求和

| 求谁 | 固定 | 要加起来的是 | 本例有几个加项 |
|------|------|--------------|----------------|
| `grad_x[i,j]` | 行 $i$，列 $j$ | 输出通道 $k=0,1$ | 2 项（$K$ 项） |
| `grad_weight[k,j]` | 通道 $k$，列 $j$ | 行 $i=0,1,2,3$ | 4 项（`NUM_ROWS` 项） |

---

### 5.4 第二件事：按代码同样的分块手算

分块策略（与前向 §3、反向内核 §6 一致）：

- **行方向：** 程序实例 0 管行 0–1，实例 1 管行 2–3；
- **列方向：** 每实例内循环 2 轮，每轮处理 2 列（`D_TILE_SIZE=2`）。

每轮每个实例手里有三块数据（形状都是 **本行块 × 本列条带**）：

| 读到什么 | 实例 0 第 0 轮 | 含义 |
|----------|----------------|------|
| `grad_output` 块 | 行 0–1，两通道 → `[[100,500],[200,600]]` | 本行块的上游梯度 |
| `x` 块 | `[[1,2],[5,6]]` | 本行块、列 0–1 |
| `weight` 条带 | 两通道、列 0–1 → `[[10,20],[50,60]]` | 全行共享，按列条带读入 |

#### 5.4.1 本列条带内算 `grad_x`（直接写入最终 `grad_x`）

对块内每个 $(i,j)$，公式与 §5.3.1 相同，只是 **$j$ 只在当前列条带里**：

```text
grad_x[0,0] = grad_output[0,0]·weight[0,0] + grad_output[0,1]·weight[1,0]
            = 10·100 + 50·500 = 26000

grad_x[0,1] = 20·100 + 60·500 = 32000
grad_x[1,0] = 10·200 + 50·600 = 32000
grad_x[1,1] = 20·200 + 60·600 = 40000
```

写成矩阵乘法（与代码思路一致）：  
`grad_x_块 = grad_output_块 @ weight_条带` → $(2\times2) @ (2\times2) = (2\times2)$。

**第 1 轮**（列 2–3，`weight` 条带 `[[30,40],[70,80]]`，`x` 块 `[[3,4],[7,8]]`）：

```text
grad_x[0,2] = 30·100 + 70·500 = 38000
grad_x[0,3] = 40·100 + 80·500 = 44000
grad_x[1,2] = 30·200 + 70·600 = 48000
grad_x[1,3] = 40·200 + 80·600 = 56000
```

实例 0 两轮合起来，**行 0–1 的 `grad_x` 已经完整**，与 §5.3.1 中对应四格 **完全一致**。  
实例 1 对行 2–3 做同样两轮，得到 §5.3.1 剩余八格。

#### 5.4.2 本列条带内算 `grad_weight`（先写部分和）

同一轮里，实例 0 **只能加行 0–1**（实例 1 管行 2–3），得到 `partial_grad_weight` 的一行：

**第 0 轮（列 0–1）：**

```text
partial_grad_weight[0, 0, 0] = grad_output[0,0]·x[0,0] + grad_output[1,0]·x[1,0]
                              = 1·100 + 5·200 = 1100

partial_grad_weight[0, 0, 1] = 2·100 + 6·200 = 1400
partial_grad_weight[0, 1, 0] = 1·500 + 5·600 = 3500
partial_grad_weight[0, 1, 1] = 2·500 + 6·600 = 4400
```

（记号 `partial_grad_weight[行块号, k, j]`。）

**第 1 轮（列 2–3）：**

```text
partial_grad_weight[0, 0, 2] = 3·100 + 7·200 = 1700
partial_grad_weight[0, 0, 3] = 4·100 + 8·200 = 2000
partial_grad_weight[0, 1, 2] = 3·500 + 7·600 = 5700
partial_grad_weight[0, 1, 3] = 4·500 + 8·600 = 6800
```

**程序实例 1**（行 2–3，`grad_output` 块 `[[300,700],[400,800]]`）两轮合计：

```text
partial_grad_weight[1, 0, :] = [7900, 8600, 9300, 10000]   # k=0，四列
partial_grad_weight[1, 1, :] = [16700, 18200, 19700, 21200]   # k=1，四列
```

**内核外按行块求和**（对应 Python `partial_grad_weight.sum(axis=0)`）：

```text
grad_weight[0,0] = 1100 + 7900 = 9000    ✓
grad_weight[0,1] = 1400 + 8600 = 10000   ✓
grad_weight[0,2] = 1700 + 9300 = 11000   ✓
grad_weight[0,3] = 2000 + 10000 = 12000  ✓
grad_weight[1,0] = 3500 + 16700 = 20200  ✓
grad_weight[1,1] = 4400 + 18200 = 22800  ✓
grad_weight[1,2] = 5700 + 19700 = 25400  ✓
grad_weight[1,3] = 6800 + 21200 = 28000  ✓
```

---

### 5.5 核心困惑：为什么有 `partial_grad_weight`，却没有 `partial_grad_x`？

你的困惑可以拆成两句：

1. **`X` 明明也按行分块了，为什么 `grad_x` 看起来一次性就算完了？**
2. **`weight` 按列条带读，`grad_weight` 却要 `partial` 再 `sum`，不对称在哪？**

#### 关键：分块有两种目的

| 目的 | 含义 |
|------|------|
| **算得更省** | 一次装不进片上 SRAM，所以 `X`、`weight` **按块读入**再算——前向、反向都这样 |
| **写结果时要不要合并** | 多个程序实例会不会 **写到同一个输出地址** |

**`grad_x`：按行分块算，但每个格子只由一个实例写入**

公式 $\partial \mathcal{L}/\partial x_{ij}$ 只依赖 **第 $i$ 行** 的 `grad_output` 和 **整列方向上的 `weight`**（通过 $k$ 求和）。  
**行号 $i$ 一旦固定，用哪几行 `grad_output` 就定了**——实例 0 永远用行 0–1，实例 1 永远用行 2–3，**互不重叠**。

因此：

- `grad_x[0,0]` **只有** 程序实例 0 会写；
- `grad_x[2,0]` **只有** 程序实例 1 会写；

列方向虽然要循环两轮，但都是在 **同一块行范围** 里往 `grad_x` 的 **不同列** 写——还是同一个实例、不同地址。  
**不需要** `partial_grad_x`：不是算出了完整 `grad_x`，而是 **每个实例负责的行块本来就可以独立写满**，直接 `store` 进最终 `grad_x` 缓冲区。

**`grad_weight`：每个 $(k,j)$ 要被所有行块各贡献一次**

公式 $\partial \mathcal{L}/\partial w_{kj}$ 要对 **$i=0,1,2,3$ 全部行** 求和。  
程序实例 0 只能算行 0–1 的部分和，实例 1 只能算行 2–3 的部分和——**两个实例都要往同一个 `grad_weight[k,j]` 贡献**，不能同时写同一地址。

所以需要：

1. 各实例写入 **`partial_grad_weight[行块号, k, j]`**（各写各的行，不冲突）；
2. 内核结束后 **`sum(axis=行块号)`** 合并成最终 `grad_weight[k,j]`。

#### 一张图总结不对称

```text
                    依赖哪些行？          多个实例写同一输出吗？
grad_x[i,j]         只要第 i 行          否 → 按行块直接写 grad_x
grad_weight[k,j]    要全部 4 行          是 → partial 再 sum
```

**`X` 的分块** 是为了 **读入和计算**（和 `weight` 按列条带读一样），**不是**因为 `grad_x` 需要跨块合并。  
真正需要「部分和 + 归约」的，是 **归约方向（求和方向）与分块方向正交** 的那一个梯度——这里是 **沿行 $i$ 求和的 `grad_weight`**。

讲义代码是 **$K=1$**：`partial_grad_weight` 形状 `(行块数, D)`，即上表去掉 $k$ 维；`grad_x` 仍按行块直接写，逻辑不变。

---

## 6. 反向内核：完整实现

> **交互图：**
> - [misc-weighted-sum-tiling-viz.html](./misc-weighted-sum-tiling-viz.html) — partial sum 与分块轴（行 vs K）
> - [misc-triton-backward-kernel-explained.html](./misc-triton-backward-kernel-explained.html) — `n_row_tiles` / `row_tile_idx` / `num_programs` 与内核 inline 注释

### 6.0 「实例 0 / 实例 1」是什么？相加的代码在哪？

读 §6.3 内核时最常见的疑惑：**手算里有实例 0、实例 1，代码里既看不见「实例」两个字，也看不见把它们加起来的 `for` 循环。**

#### 实例 = `tl.program_id(0)`

| 手算里的叫法 | 代码里的名字 | 本例（4 行，tile 高 2） |
|--------------|--------------|-------------------------|
| 程序实例 0 | `row_tile_idx = tl.program_id(0)` 为 **0** | 负责 `X` 行 0–1 |
| 程序实例 1 | `row_tile_idx = tl.program_id(0)` 为 **1** | 负责 `X` 行 2–3 |

**怎么冒出两份实例？** 在 Python 里 **launch grid** 决定份数：

```python
n_row_tiles = triton.cdiv(n_rows, ROWS_TILE_SIZE)   # 本例：cdiv(4,2)=2
weighted_sum_backward[(n_row_tiles,)](...)            # grid = (2,) → 并行 2 个 program
```

方括号 `(2,)` 表示：同一份 `weighted_sum_backward` 内核在 GPU 上 **同时启动 2 次**。  
每一次里 `tl.program_id(0)` 不同（0 或 1），于是 `offsets=(row_tile_idx * ROWS_TILE_SIZE, …)` 指向 **不同的行块**。

**没有「实例 2」：** 4 行、每块 2 行 → 只有 `program_id(0) ∈ {0, 1}`。行数很大时会有 0, 1, 2, …, `n_row_tiles-1`。

#### 实例 0 + 实例 1 在哪相加？——不在内核里，在 Python

内核里 **每个 program 只写自己那一行** 的 `partial_grad_weight`：

```text
partial_grad_weight 形状 (n_row_tiles, D) = (2, 4)

program_id(0)=0  →  写入 partial_grad_weight[0, :]   （行 0–1 的部分和）
program_id(0)=1  →  写入 partial_grad_weight[1, :]   （行 2–3 的部分和）
```

两块写 **不同地址**，所以内核里 **不需要**、也 **没有** `partial[0] + partial[1]`。

**合并发生在内核结束之后**，`WeightedSumFunc.backward` 里这一行（§6.5）：

```python
grad_weight = partial_grad_weight.sum(axis=0)
```

`axis=0` 沿 **行块维**（就是 `program_id` 那一维）把实例 0、实例 1 的贡献加起来，得到最终 `grad_weight[j]`。  
对应 §5.4.2 手算：`grad_weight[0,0] = 1100 + 7900 = 9000`。

```text
┌─────────────────────────────────────────────────────────────┐
│  weighted_sum_backward[(2,)]   ← 2 个 program 并行          │
│    program_id=0: store → partial_grad_weight[0, :]        │
│    program_id=1: store → partial_grad_weight[1, :]        │
│    （各自 store → grad_x 的不同行，互不冲突）                 │
└─────────────────────────────────────────────────────────────┘
                              ↓ 内核返回后，仍在 CPU/GPU 上
┌─────────────────────────────────────────────────────────────┐
│  grad_weight = partial_grad_weight.sum(axis=0)   ← 这里相加  │
└─────────────────────────────────────────────────────────────┘
```

**`grad_x` 为什么内核里也不用加？** 每个 `program_id` 写的是 `grad_x` 的 **不同行**，和 `partial_grad_weight` 各写各的行块一样——但 `grad_x` 行块已经是 **最终结果**，不需要再沿行块归约，所以 **没有** `partial_grad_x`，也 **没有** 第二行 `sum`。

#### 那 grad_x 能不能也 forced 出 partial sum？

**可以，但要换「按什么切 program」，不是行尾没对齐。**

先记住 §5.2 里 `grad_x[i,j]` 的公式（weight 有 2 行时）：

```text
grad_x[i,j] = grad_output[i,0]×weight[0,j] + grad_output[i,1]×weight[1,j]
              └─ 第一项 ─────────────┘   └─ 第二项 ─────────────┘
```

这是 **两个数相加**。第一项只用 weight **第 0 行** 和 grad_output **第 0 列**；第二项只用 weight **第 1 行** 和 grad_output **第 1 列**。

| 怎么切 program | grad_x | grad_weight |
|----------------|--------|-------------|
| **按 X 的行**（讲义） | 每个 program 管 X 的几行；算 `grad_x[i,j]` 时两项都在手边 → **一次算完，直写** | 每个 `weight[j]` 要加 **所有 X 行** → **partial + sum** |
| **按 weight 的行**（假想） | program 0 只算第一项，program 1 只算第二项；**同一** `grad_x[i,j]` 被拆成两半 → **partial + sum** | 每个 program 包办 weight 的 **一整行**、内部扫完所有 X 行 → **直写** |

行数不是 tile 整数倍（5 行、tile=2）时，仍是「每个 `grad_x[i,j]` 只归一个 program」→ **仍不需要** `partial_grad_x`。  
要对 `grad_x` 做 partial，必须 **故意** 让两个 program 各算上面公式里的一项（逐步推导见 [misc-weighted-sum-tiling-viz.html](./misc-weighted-sum-tiling-viz.html) 标签页 B）。

```text
假想 program 0：partial_grad_x[0,i,j] = grad_output[i,0] × weight[0,j]
假想 program 1：partial_grad_x[1,i,j] = grad_output[i,1] × weight[1,j]
Python：grad_x[i,j] = partial_grad_x[0,i,j] + partial_grad_x[1,i,j]
```

讲义选 **按 X 的行** 切，是因为前向也是按行 tile；不是 grad_x **算不出来** partial，而是这种切法 **不需要**。

### 6.1 与 §5.4、§5.5 的对应关系

| §5 手算 | §6 代码（$K=1$） |
|---------|------------------|
| 程序实例 0 / 1 | `row_tile_idx = tl.program_id(0)`；launch `[(n_row_tiles,)]` |
| `grad_output` 块 `[[100,500],[200,600]]` | 一维 `grad_output`，长度 `ROWS_TILE_SIZE` |
| `weight` 条带 `[10,20]` | 一维 `weight`，长度 `D_TILE_SIZE` |
| `grad_x` 块 = `grad_output[:, None] * weight[None, :]` | 广播乘法；$K>1$ 时等价于小矩阵乘 |
| `partial_grad_weight[行块, k, j]` | `partial_grad_weight[行块, j]`（$K=1$）；行块 = `program_id(0)` |
| `1100 + 7900 = 9000` | **`grad_weight = partial_grad_weight.sum(axis=0)`**（§6.5，内核外） |
| `grad_x` 直接 `store` | `grad_x_block_ptr` 直接写最终 `grad_x`，无 partial |


### 6.2 五个块指针：反向比前向多盯谁

与前向一样按 **行** 分块、沿 **$D$（列）** 循环；额外引入 **`grad_output`** 和 **`partial_grad_weight`**。  
（$K=1$ 时 `grad_output` 是一维；§5 的 $K=2$ 只是多一列，块指针逻辑相同。）

| 块指针 | 逻辑形状（$K=1$ 代码） | 本实例一次处理 | 作用 |
|--------|------------------------|----------------|------|
| `grad_output_block_ptr` | `(NUM_ROWS,)` | `ROWS_TILE_SIZE` 个标量 | 读本行块的上游梯度 |
| `x_block_ptr` | `(NUM_ROWS, D)` | `(ROWS_TILE_SIZE, D_TILE_SIZE)` | 读前向时的 `x` 本行块、本列条带 |
| `weight_block_ptr` | `(D,)` | `D_TILE_SIZE` | 读 `weight` 当前列条带（全行共享） |
| `grad_x_block_ptr` | `(NUM_ROWS, D)` | `(ROWS_TILE_SIZE, D_TILE_SIZE)` | **直接写最终** `grad_x`（§5.5：无跨实例冲突） |
| `partial_grad_weight_block_ptr` | `(n_row_tiles, D)` | `(1, D_TILE_SIZE)` | 写本实例对 `grad_weight` 的 **部分和**（§5.5：跨行块要合并） |

`partial_grad_weight_block_ptr` 的 `offsets=(row_tile_idx, 0)`：`row_tile_idx` 就是 **`tl.program_id(0)`**，每个 program 写 **自己那一行块号** 对应的部分和，实例之间 **不写同一地址**。

### 6.3 完整 `weighted_sum_backward` 内核

```python
@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,
    grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwb, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    # ← 「实例 0 / 实例 1」：本 program 在 launch grid 第 0 维上的编号
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)   # 与 Python 里 grid 长度一致，本例 = 2

    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),   # program_id=0 → 行 0 起；=1 → 行 2 起
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),   # 各 program 写 grad_x 的不同行
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D), strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),   # program_id=0 → 第 0 行；=1 → 第 1 行（部分和缓冲区）
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0),
    )

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # §5.4.1（K=1）：grad_x[i,j] = grad_output[i] * weight[j]
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # §5.4.2（K=1）：partial_grad_weight[行块,j] += sum_{i in 本块} grad_output[i] * x[i,j]
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
    # 注意：内核到此结束。没有 partial[0]+partial[1]；合并见 §6.5 的 sum(axis=0)
```

### 6.4 手算对照

§5.4 已用 **同一组数字** 走完分块全过程。读代码时记三处：

| 看什么 | 在哪 |
|--------|------|
| 有几个「实例」 | Python：`weighted_sum_backward[(n_row_tiles,)]`，本例 `[(2,)]` |
| 当前是实例几 | 内核：`row_tile_idx = tl.program_id(0)` → 0 或 1 |
| 实例 0 + 1 相加 | **内核外** Python：`grad_weight = partial_grad_weight.sum(axis=0)` |

### 6.5 `backward` 静态方法：接回 §4 的计算图

```python
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # ... 见 §4.3 ...

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D),   # 形状 (2, 4)：第 0 维 = program_id
            device=x.device, dtype=x.dtype,
        )
        grad_x = torch.empty_like(x)

        # grid (2,) → 启动 2 个 program；各写 partial_grad_weight 的一行、grad_x 的不同行
        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        # ← 实例 0 与 实例 1 在这里合并（不在 Triton 内核里）
        grad_weight = partial_grad_weight.sum(axis=0)
        return grad_x, grad_weight
```

数据流闭合：

```text
loss.backward()
  → PyTorch 调 WeightedSumFunc.backward(grad_out)
  → 分配 grad_x、partial_grad_weight（形状含 program 维）
  → launch weighted_sum_backward[(n_row_tiles,)]  # 多 program 并行，各 program_id 不同
  → grad_weight = partial_grad_weight.sum(axis=0)  # 内核外合并 partial
  → 返回 (grad_x, grad_weight)，继续向更早的层传播
```

### 6.6 对外接口与 `grad_fn`

```python
f_weightedsum = WeightedSumFunc.apply
```

调用示例：

```python
y = f_weightedsum(x, weight)
# y 例如：tensor([90.8563, -93.6815, ...], device='cuda:0',
#              grad_fn=<WeightedSumFuncBackward>)
```

`grad_fn=<WeightedSumFuncBackward>` 表示：该张量在计算图上挂着我们实现的反向；`loss.backward()` 时会沿此节点继续调用 §6.5 的 `backward`。**加权求和的 Triton 前向 + 反向 + autograd 闭环到此完成。**

---

## 7. 小结与下一步：Flash Attention 第二版前向

加权求和教的是 **通用套路**：

1. 用 Triton 写分块前向 / 反向；
2. 用 `autograd.Function` 保存反向所需张量、启动内核；
3. 大归约拆成 **块内部分和 + 块外 `sum`**（仅当多个块要写同一输出时，见 §5.5）。

**Flash Attention 第二版前向** 在同一套思想上做三件事（下一篇学习文档展开）：

| 技术 | 解决什么问题 |
|------|-------------|
| **分块（tiling）** | 不把完整注意力矩阵 `P`（形状含 `sequence_length²`）写入高带宽内存（HBM） |
| **重算（recomputation）** | 反向只存检查点（如对数和向量 `L`），需要时再算 `P` |
| **算子融合（operator fusion）** | 前向尽量在一个内核里完成，减少 HBM 与片上 SRAM 之间的往返 |

朴素注意力前向：$S = QK^\top/\sqrt{d}$ → $\mathrm{softmax}$ → $O = PV$；反向依赖巨大的 $P$，长序列时显存与访存都是瓶颈——这正是接下来要用 Triton 实现 Flash Attention 的原因。

---

## 8. 对照表：加权求和 vs 即将写的注意力

| | 加权求和（本节） | Flash Attention（下一节） |
|--|------------------|---------------------------|
| 分块维 | 行 × 特征维 `D` | 查询块 × 键块（序列维） |
| 难点 | `grad_weight` 沿行求和 → 跨行块要 `partial` + `sum`；`grad_x` 按行块直接写 | 在线 softmax、保存 `L`、反向重算 `P` |
| 目标 | 学会 Triton + autograd | 长序列下访存与峰值显存不再 ∝ `sequence_length²` |

---

## 9. 术语表

读讲义 PDF、本文和代码时，**同一概念经常有两个名字**（尤其 `D` 和 `H`）。下表按「你实际会看到什么」列出来，并用手算例子的具体数字对应。

### 9.1 张量形状：谁是谁

| 术语 | 在哪出现 | 什么意思 | 手算例子（§3.3 / §5） | 和别的符号的关系 |
|------|----------|----------|------------------------|------------------|
| **`n` / `n_rows` / `NUM_ROWS`** | 讲义 `n×h`；代码 `x.shape[0]`、`NUM_ROWS` | **X 有多少行**（要算多少个输出位置） | `4`（行 0–3） | 与列数无关 |
| **`D`** | **代码里几乎都用这个**：`x.shape[1]`、`weight` 长度、`partial_grad_weight` 第 2 维 | **X 有多少列**；也是 **weight 向量的长度**（每个输出是「一行 X」与 weight 的内积） | `4`（列 0–3；`weight = [10,20,30,40]` 长度 4） | **= 讲义里的 `H` / `h`**（见下行） |
| **`H` / `h`** | 讲义 PDF 第 22 页：`partial_grad_weight` 是 `n_row_tiles × H`；§2.1 写 `x ∈ ℝ^{n×h}` | **和 `D` 完全同一个数**：特征维 / 列数 / weight 长度 | `H = 4`，就是 X 的 4 列 | **不是** attention 里的 head 数；**不是** HBM 的 H |
| **`K`** | §5 推广：weight 有 `K` 行时，输出有 `K` 列 | **weight 矩阵有多少行**；前向时 X 的每一行要算 `K` 次内积，得到 `K` 个数 | 讲义代码 `K=1`（weight 一行）；§5 手算 `K=2`（weight 两行 → `y` 两列） | 讲义代码里 **没有单独的 `K` 变量**，因为固定 `K=1` |
| **`weight[j]` / `w_j`** | 公式、一维 weight | weight 的**第 j 个分量**（对应 X 的**第 j 列**） | `weight[0]=10` 乘 `x[i,0]` | `j` 从 `0` 到 `D-1` |
| **`x[i,j]`** | 手算表、梯度下标 | X **第 i 行、第 j 列**的元素 | `x[0,0]=1`，`x[1,2]=7` | `i ∈ [0, n_rows)`，`j ∈ [0, D)` |
| **`y[i]`**（K=1） | §3.3 前向 | 第 i 行的加权求和结果（**一个标量**） | `y[0]=300` | `y` 形状 `(n_rows,)` |
| **`y[i,k]` / `grad_output[i,k]`**（K≥2） | §5 | 第 i 行、用 weight **第 k 行**算出的那个输出；反向时上游传来的梯度 | `grad_output[0,0]=100`，`grad_output[0,1]=500` | `k ∈ [0, K)`；**k 就是「weight 的第几行 / y 的第几列」** |

**关于 `D` 和 `H`：** 确实扯淡——**是同一个东西的两个记号**。讲义 PDF 用 **`H`/`h`**（习惯上表示 hidden / feature size）；本仓库 **Triton 内核和 PyTorch 代码统一写 `D`**（dimension）。看到 `n_row_tiles × H` 就当成 `partial_grad_weight.shape == (n_row_tiles, D)`，不要多想。

---

### 9.2 分块与并行：tile、program、grid

| 术语 | 在哪出现 | 什么意思 | 手算例子 |
|------|----------|----------|----------|
| **tile / 分块** | §3.2 起 | 把大矩阵切成小块，每次只处理一小块 | 4×4 的 X 切成四块 2×2 |
| **`ROWS_TILE_SIZE`** | 内核 `constexpr`；`ctx.ROWS_TILE_SIZE` | **行方向**每个 tile 多高（每个 program 管几行 X） | 手算 `2`；真实训练代码里常是 `16` |
| **`D_TILE_SIZE`** | 内核 `constexpr` | **列方向**每次循环处理几列（沿 `D` 切条带） | 手算 `2`（4 列要循环 2 轮） |
| **`n_row_tiles`** | Python：`triton.cdiv(n_rows, ROWS_TILE_SIZE)`；PDF 第 22 页 | **行方向一共切了几块** = **launch 了几个 program**（计数，不是某一行下标） | `cdiv(4,2)=2` |
| **`row_tile_idx`** | 内核：`tl.program_id(0)` | **当前 program 是第几块行 tile**（从 0 开始） | `0` 管行 0–1，`1` 管行 2–3 |
| **`program_id(0)`** | Triton API | 与 `row_tile_idx` **同一件事**：本 program 在 grid 轴 0 上的编号 | 实例 0、实例 1 |
| **launch grid `[(n_row_tiles,)]`** | `weighted_sum_*[(...)]` | 启动 **`n_row_tiles` 份**并行 program；方括号里是 grid 形状 | `[(2,)]` → 2 个 program 同时跑 |
| **`tl.num_programs(0)`** | 反向内核内 | 读回 grid 轴 0 上有几个 program；**恒等于** `n_row_tiles` | 用来声明 `partial_grad_weight` 第 0 维长度 |

---

### 9.3 反向与 partial buffer

| 术语 | 在哪出现 | 什么意思 | 手算例子 |
|------|----------|----------|----------|
| **`grad_output`** | `backward(ctx, grad_output)` | 上游传下来的梯度；形状与 **前向输出 `y`** 相同 | K=1：长度 4 的向量；K=2：4×2 矩阵 |
| **`grad_x`** | 对 `x` 的梯度 | 形状与 **`x` 相同** `(n_rows, D)` | `grad_x[0,0]=26000`（§5，K=2） |
| **`grad_weight`** | 对 `weight` 的梯度 | K=1：形状 `(D,)`；K=2：形状 `(K, D)` | `grad_weight[0,0]=9000` |
| **`partial_grad_weight`** | Python 分配；内核写入 | **每个行块**先算出的 `grad_weight` 一部分；形状 **`(n_row_tiles, D)`**（K=1）或 **`(n_row_tiles, K, D)`**（K=2） | 实例 0 写出 `[1100,1400,…]`，实例 1 写出 `[7900,8600,…]` |
| **`partial_grad_weight.sum(axis=0)`** | `WeightedSumFunc.backward` | 沿 **行块维** 相加，得到最终 `grad_weight` | `1100+7900=9000`（第 0 列） |
| **`partial_grad_x`** | §5.5 假想情形 | 讲义 **实际代码没有**；若按 weight 的行切 program，同一 `grad_x[i,j]` 被拆成多项再合并时才需要 | 见 [misc-weighted-sum-tiling-viz.html](./misc-weighted-sum-tiling-viz.html) 标签 B |
| **直写** | §5.5 | 一个 program 算完就直接 `store` 到**最终**缓冲区，不经 partial | 讲义里 **`grad_x` 直写**；按行切时 **`grad_weight` 不直写** |
| **partial + sum** | §5.5、PDF 第 22 页 | 多个 program 各写一部分到 partial 张量，**内核外** `sum` 合并 | `grad_weight = partial_grad_weight.sum(axis=0)` |

**一眼记：** 讲义按 **X 的行** 切 program → **`grad_x` 直写**（各行归不同 program），**`grad_weight` 要 partial**（同一列要加所有行）。若改成按 **weight 的行** 切，角色对调（§5.5）。

---

### 9.4 内存与 PyTorch 衔接（易混）

| 术语 | 什么意思 | 别和什么搞混 |
|------|----------|--------------|
| **`HBM`** | GPU **片外**高带宽内存（显存） | 字母 H 在这里是 **High**，和特征维 **`H`/`h` 无关** |
| **`stride`** | 沿某一维走 1 格，在扁平内存里跳几个元素 | `x.stride(0)` 常是 `D`（行优先） |
| **`block_ptr` / 块指针** | `tl.make_block_ptr`：描述「从哪读/写多大一块」 | 不是 PyTorch 的 `Tensor` |
| **`autograd.Function`** | 自定义前向/反向，挂到计算图 | `grad_fn=<WeightedSumFuncBackward>` 就来自这里 |
| **`f_weightedsum`** | `WeightedSumFunc.apply` 的别名 | 用法同普通函数，但走自定义反向 |

---

### 9.5 符号对照（复制用）

```text
讲义 PDF          本文 / 代码              手算 4×4 例子
─────────────────────────────────────────────────────────
n, H              n_rows, D               4 行, 4 列
w ∈ ℝ^H           weight.shape == (D,)    [10,20,30,40]
n_row_tiles × H   (n_row_tiles, D)        (2, 4) 的 partial_grad_weight
K（§5 推广）      weight 行数 / y 列数     K=1 或 K=2
program 实例 i    program_id(0)           0 或 1
```

---

## 10. 讲义收尾段落在说什么？（`f_weightedsum` 与 `grad_fn`）

讲义 PDF 第 22 页末尾原文大意：

```text
f_weightedsum = WeightedSumFunc.apply
调用 f_weightedsum(x, w) 会得到类似：
tensor([90.8563, -93.6815, ...], device='cuda:0', grad_fn=<WeightedSumFuncBackward>)
注意 grad_fn —— 说明 PyTorch 知道反向时该调谁。
加权求和的 Triton 实现到此完成。
```

下面逐项拆开：**是什么、从哪来、为什么要提它**。

### 10.1 `f_weightedsum` 是什么

```python
f_weightedsum = WeightedSumFunc.apply
```

| 名字 | 是什么 |
|------|--------|
| `WeightedSumFunc` | 你写的 `torch.autograd.Function` 子类（§4、§6） |
| `.apply` | PyTorch 规定的**唯一正确入口**：会自动建计算图节点、调 `forward`、在输出上挂 `grad_fn` |
| `f_weightedsum` | 只是给 `.apply` 起的**短别名**，用起来像普通函数 |

**不是** `torch.nn.functional` 里自带的算子；是 **你自己实现、自己注册进 autograd** 的加权求和。

调用方式：

```python
y = f_weightedsum(x, weight)   # 讲义里第二个参数写作 w，就是 weight
```

- **`x`**：输入矩阵（可有 batch 维，内部会展平成行×`D`）
- **`weight` / `w`**：长度 `D` 的权重向量（讲义符号 `w ∈ ℝ^H`，和代码里 `weight` 同一物）

前向做的事（§4.3）：整理形状 → GPU 上分配 `y` → launch `weighted_sum_fwd` → 返回 `y`。

### 10.2 打印出来的 `tensor([90.8563, -93.6815, ...])` 是什么

这是 **前向算出来的输出 `y`**，不是梯度，也不是错误信息。

| 打印字段 | 含义 |
|----------|------|
| 一维数字列表 | `y` 的每个元素 = `x` 的**一行**与 `weight` 做加权求和后的**一个标量** |
| `90.8563, -93.6815, ...` | 讲义用**随机初始化的** `x`、`weight` 跑出来的具体数值；**和手算 §3.3 的 300、700 无关**，只是格式示例 |
| `...` | 中间还有很多元素，终端省略显示 |
| `device='cuda:0'` | 张量在 **第 0 块 GPU** 上；因为 Triton 内核跑在 CUDA 上 |

**形状：** 若 `x` 展平后有 `n_rows` 行，则 `y` 长度 = `n_rows`（K=1 时）。  
§3.3 手算若调用，应得到 `tensor([300., 700., 1100., 1500.])`。

### 10.3 `grad_fn=<WeightedSumFuncBackward>` 是什么

`grad_fn` = **gradient function**，挂在这个张量上的 **「反向回调」指针**。

前向 `WeightedSumFunc.forward` 返回 `y` 时，PyTorch 会在 `y` 上记录：

```text
这个 y 是怎么来的？
→ 来自 WeightedSumFunc.forward(x, weight)
→ 反向时请调 WeightedSumFunc.backward
```

所以打印里显示 `grad_fn=<WeightedSumFuncBackward>`（PyTorch 自动生成的 backward 包装类名）。

**它本身不算梯度**；只表示：**`y` 在计算图里占了一个节点，且这个节点知道怎么反传**。

若你写：

```python
y = f_weightedsum(x, weight)
loss = y.sum()
loss.backward()
```

PyTorch 会沿图往回走，到 `y` 这一站时调用 §6.5 的 `backward` → launch `weighted_sum_backward` → 得到 `grad_x`、`grad_weight`。

若没有 `grad_fn`（例如 `y = x.detach()` 或 `with torch.no_grad()`），`y` 就不在图上，`loss.backward()` **不会**进你的 Triton 反向内核。

### 10.4 「PyTorch knows what to call in the backward pass」是什么意思

直译：**当这个张量出现在计算图里时，PyTorch 知道反向该调哪段代码。**

具体就是：

1. **前向**：`f_weightedsum(x, weight)` → 执行 Triton 前向内核 → 返回 `y`，并 `save_for_backward(x, weight)`
2. **建图**：`y.grad_fn` 指向 `WeightedSumFuncBackward`
3. **反向**：用户调 `loss.backward()` → PyTorch 把 ∂loss/∂y 传给 `WeightedSumFunc.backward` → Triton 反向内核 → 返回 `grad_x`, `grad_weight` → 继续传给更前面的层

**Triton 内核自己不会参与 autograd**；是 `WeightedSumFunc` 在 Python 里把「launch 内核」嵌进 PyTorch 规定的前向/反向接口里，PyTorch 才能在 `backward()` 时找到你。

### 10.5 「This completes our Triton implementation」是什么意思

意思是：**到这一步，加权求和这条链路已经闭环**，不再缺零件：

| 已完成 | 对应章节 |
|--------|----------|
| Triton **前向**内核 `weighted_sum_fwd` | §3 |
| Triton **反向**内核 `weighted_sum_backward` | §6 |
| **`partial_grad_weight` + `sum(axis=0)`** 合并 | §5.5、§6.5 |
| **`autograd.Function`** 接上 `forward` / `backward` | §4、§6 |
| **对外可调用的函数** `f_weightedsum` | §6.6 |

之后你可以像用 `torch.matmul` 一样，在更大模型里写 `y = f_weightedsum(x, weight)`，训练时 `loss.backward()` 会自动用到你的 GPU 实现。

**还没做、也不在这句话范围里的：** Flash Attention（讲义下一节 §4.2.2）、和 `torch.nn.functional` 官方 API 名字对齐等——那些是新任务。

### 10.6 最小可运行片段（对照讲义打印）

```python
import torch
from your_module import WeightedSumFunc   # 讲义里的完整实现

f_weightedsum = WeightedSumFunc.apply

x = torch.randn(8, 64, device="cuda")      # 8 行，D=64
weight = torch.randn(64, device="cuda")  # 长度 64

y = f_weightedsum(x, weight)
print(y)
# tensor([..., ...], device='cuda:0', grad_fn=<WeightedSumFuncBackward>)
#          ↑ 8 个数，每个是一行 x 与 weight 的内积

print(y.grad_fn)                           # <WeightedSumFuncBackward object at 0x...>
print(y.requires_grad)                     # False（y 是输出；梯度在 x、weight 上通过 backward 填）

loss = y.sum()
loss.backward()
print(x.grad.shape)                        # 与 x 相同
print(weight.grad.shape)                   # (64,)
```

**三句话总结讲义那段：**

1. **`f_weightedsum(x, w)`** = 用你写的 Triton 前向算加权求和，得到输出向量 `y`。  
2. **`grad_fn`** = `y` 上贴的纸条：「我来自 `WeightedSumFunc`，反传时请调它的 `backward`」。  
3. **「实现完成」** = 前向内核 + 反向内核 + autograd 包装 + 可调用的 `f_weightedsum` 全都齐了，能放进训练循环。

