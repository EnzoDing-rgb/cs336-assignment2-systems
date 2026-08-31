# FSDP 的通信-计算权衡：以单个 FFN 层为例

> 对应 handout 的 `fsdp_calcs` 问题。设定与 [data-parallel-calculations.md](./data-parallel-calculations.md) 相同：单层 FFN、全局 batch $B$、隐藏维 $D$、FFN 宽维 $D_{\mathrm{FF}}$、FP16（每元素 2 字节）、环形 collective、每台算力 $C$（FLOP/s）、出向带宽 $W$（字节/s）。区别是并行方式从「切 batch、参数全量复制」换成 **FSDP**：参数也按 $N_{\mathrm{FSDP}}$ 分片，前向 all-gather 权重，反向 reduce-scatter 梯度。

---

## 0. 先建立直觉：FSDP 比 DDP 多付了什么

数据并行（DDP）的账你已经算过：每台设备常驻**完整**的 $W_1,W_2,W_3$，batch 切成 $B/N$；**前向零通信**，**反向**对三个权重梯度做一次 all-reduce。

FSDP（ZeRO-3 风格）换了一套交易：

| | DDP | FSDP |
|--|-----|------|
| 参数存储 | 每卡完整 $W_1,W_2,W_3$ | 每卡只存 $1/N_{\mathrm{FSDP}}$ 分片 |
| 前向 | 直接 matmul | **先 all-gather 拼完整权重，再 matmul** |
| 反向权重梯度 | all-reduce（每人拿完整 $\mathrm d W$） | **reduce-scatter**（每人只留自己的 $\mathrm d W$ 分片） |
| 反向激活梯度 | 本地 | 本地（与 DDP 相同） |
| 显存 | 参数 × $N$ 份 | 参数 ÷ $N_{\mathrm{FSDP}}$ |

**FSDP 用通信换显存。** 计算 matmul 的 FLOP 数不变——因为 all-gather 之后，每台设备看到的仍是完整权重，做的仍是同样的矩阵乘。变的是**什么时候、传什么、传多少**。

下面按 handout 的三问 (a)(b)(c) 展开；每一问都给出**前向**和**反向**两个答案，并各附一句理由（可直接抄进作业 deliverable）。

---

## 符号表

| 符号 | 含义 |
|------|------|
| $B$ | 全局 batch |
| $D$ | $d_{\mathrm{model}}$ |
| $D_{\mathrm{FF}}$ | FFN 中间维 |
| $N_{\mathrm{FSDP}}$ | FSDP 设备数（同时切 batch + 切参数） |
| $C$ | 单设备算力（FLOP/s） |
| $W$ | 单设备出向带宽（字节/s） |
| $S$ | 本层三个权重矩阵的总字节数：$S = 6\,D\,D_{\mathrm{FF}}$（FP16） |
| $W_1,W_2,W_3$ | 形状 $(D,D_{\mathrm{FF}})$、$(D,D_{\mathrm{FF}})$、$(D_{\mathrm{FF}},D)$ |

FFN 前后向公式与 [data-parallel-calculations.md §1](./data-parallel-calculations.md) 相同，此处不重复推导。

---

## 1. FSDP 下这一层在干什么

batch 仍按 $N_{\mathrm{FSDP}}$ 切：每台设备上的有效 batch 是 $B/N_{\mathrm{FSDP}}$。三个权重矩阵各切成 $N_{\mathrm{FSDP}}$ 份，每台只**常驻**自己那一份。

**前向**（每层三个 matmul，各需一次 all-gather）：

```text
对 W1:  all-gather → x·W1  → x1
对 W2:  all-gather → x·W2  → x2
        z = f(x1) ⊙ x2
对 W3:  all-gather → z·W3  → y
```

**反向**（激活梯度本地算；权重梯度本地算完后 reduce-scatter）：

```text
dy 已知（与 DDP 相同，每卡相同）
对 W3:  本地算 dW3、dz  → reduce-scatter(dW3)
逐元素反传
对 W1,W2: 本地算 dW1,dW2,dx  → reduce-scatter(dW1), reduce-scatter(dW2)
```

实现细节（预取、cast、RMSNorm 不切分等）见 [distributed-communication.md §9](./distributed-communication.md) 与 `cs336_systems/distributed/fsdp.py`；**本题只计 FFN 三个 Linear 的权重通信**，与 handout 一致。

---

## 2. (a) 计算量：FLOPs

### 关键结论

FSDP **不改变 matmul 次数**，只改变权重从哪来。all-gather 是内存搬运，不计入 FLOP。

| 阶段 | 每设备 FLOPs | 一句话理由 |
|------|-------------|-----------|
| **前向** | $\displaystyle \frac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$ | 3 个 matmul，每个 $2\cdot(B/N_{\mathrm{FSDP}})\cdot D\cdot D_{\mathrm{FF}}$，与 DDP 前向相同。 |
| **反向** | $\displaystyle \frac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$ | 6 个 matmul（3 激活梯度 + 3 权重梯度），每个 $2\cdot(B/N_{\mathrm{FSDP}})\cdot D\cdot D_{\mathrm{FF}}$，与 DDP 反向相同。 |

### 为什么和 DDP 一模一样

把 DDP 想成「参数本来就在本地」；FSDP 是「参数分片在本地，算之前临时拼成完整的」。拼好之后，$x\cdot W_1$ 的 FLOP 计数与未分片时相同。全部设备加总的 FLOP 仍是 $6BD D_{\mathrm{FF}}$（前向）和 $12BD D_{\mathrm{FF}}$（反向），与 $N_{\mathrm{FSDP}}$ 无关——**并行效率在计算侧不损失**，和 DDP 一样。

---

## 3. (b) 通信时间

### 3.1 传多少字节

三个权重矩阵元素总数 $3\cdot D\cdot D_{\mathrm{FF}}$；FP16 下

$$
S = 6\,D\,D_{\mathrm{FF}} \quad \text{（字节）}.
$$

- **前向**：每个 matmul 前 all-gather **一个**完整权重矩阵（大小 $2DD_{\mathrm{FF}}$ 字节）→ 共 **3 次 all-gather**，合计 $S$ 字节「被拼出来一次」。
- **反向**：每个权重梯度做 **1 次 reduce-scatter**（同样大小）→ 共 **3 次 reduce-scatter**，合计 $S$ 字节。

环形算法下，单次 all-gather 或 reduce-scatter 的用时为 $\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}}\cdot\frac{S_{\mathrm{单次}}}{W}$（推导见 [alternate-ring-all-reduce.md §2](./alternate-ring-all-reduce.md)）。三次叠加：

$$
T_{\mathrm{comm}} = \frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}} \cdot \frac{6\,D\,D_{\mathrm{FF}}}{W}.
$$

**前向和反向的公式形式相同**，只是发生的阶段不同。

| 阶段 | 每设备通信时间 | 一句话理由 |
|------|---------------|-----------|
| **前向** | $\displaystyle T_{\mathrm{comm,fwd}} = \frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}} \cdot \frac{6\,D\,D_{\mathrm{FF}}}{W}$ | 三个权重各需一次 all-gather，环形 all-gather 每设备出向量 $\frac{N-1}{N}$ 倍于单次矩阵字节数，三次合计 $\frac{N-1}{N}\cdot 6DD_{\mathrm{FF}}$。 |
| **反向** | $\displaystyle T_{\mathrm{comm,bwd}} = \frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}} \cdot \frac{6\,D\,D_{\mathrm{FF}}}{W}$ | 三个权重梯度各需一次 reduce-scatter，环形 reduce-scatter 每设备出向量同样为 $\frac{N-1}{N}\cdot 6DD_{\mathrm{FF}}$。 |

### 3.2 和 DDP 比：系数 2 去哪了

DDP **反向**用的是 **all-reduce**（reduce-scatter + all-gather 两段），所以通信时间前有因子 **2**：

$$
T_{\mathrm{comm,DDP,bwd}} = \frac{2(N_{\mathrm{DP}}-1)}{N_{\mathrm{DP}}} \cdot \frac{6\,D\,D_{\mathrm{FF}}}{W}.
$$

FSDP 的单程 collective（all-gather **或** reduce-scatter）只有 **一段**，因子是 1 而不是 2。

但要注意：**FSDP 前向也要通信，DDP 前向没有。** 一整步（前向 + 反向）FSDP 的权重相关通信量，在环形模型下大约是

$$
\underbrace{\frac{N-1}{N}\cdot 6DD_{\mathrm{FF}}}_{\text{前向 all-gather}} + \underbrace{\frac{N-1}{N}\cdot 6DD_{\mathrm{FF}}}_{\text{反向 reduce-scatter}} = \frac{2(N-1)}{N}\cdot 6DD_{\mathrm{FF}},
$$

与 DDP 一步的 all-reduce **总量同级**——FSDP 不是「通信更少」，而是「把同样量级的通信拆进了前向和反向，换显存」。

---

## 4. (c) 什么时候通信成为瓶颈

假设计算与通信可重叠，瓶颈判据：$T_{\mathrm{comm}} \ge T_{\mathrm{compute}}$。

每设备计算时间：

$$
T_{\mathrm{compute,fwd}} = \frac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}\,C}, \qquad
T_{\mathrm{compute,bwd}} = \frac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}\,C}.
$$

### 前向临界点

令 $T_{\mathrm{comm,fwd}} \ge T_{\mathrm{compute,fwd}}$，约去 $6DD_{\mathrm{FF}}$：

$$
\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}\,W} \ge \frac{B}{N_{\mathrm{FSDP}}\,C}
\quad\Longrightarrow\quad
N_{\mathrm{FSDP}} \ge 1 + B\cdot\frac{W}{C}.
$$

| | 答案 | 一句话理由 |
|--|------|-----------|
| **前向** | $N_{\mathrm{FSDP}} \ge 1 + B\cdot W/C$ | 前向 all-gather 的通信量与每设备 matmul 计算量打平时，通信成为瓶颈；解不等式得此阈值。 |

### 反向临界点

令 $T_{\mathrm{comm,bwd}} \ge T_{\mathrm{compute,bwd}}$，约去 $6DD_{\mathrm{FF}}$：

$$
\frac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}\,W} \ge \frac{2B}{N_{\mathrm{FSDP}}\,C}
\quad\Longrightarrow\quad
N_{\mathrm{FSDP}} \ge 1 + 2B\cdot\frac{W}{C}.
$$

| | 答案 | 一句话理由 |
|--|------|-----------|
| **反向** | $N_{\mathrm{FSDP}} \ge 1 + 2B\cdot W/C$ | 反向计算量是前向 2 倍而通信量相同，故需约 2 倍大的 $B$ 或更小的 $N_{\mathrm{FSDP}}$ 才不被通信拖垮；解不等式得此阈值。 |

### 读这两个阈值

1. **$D$ 和 $D_{\mathrm{FF}}$ 再次完全消失**——与 DDP 一样，FLOP 与字节数同比例于 $DD_{\mathrm{FF}}$，比瓶颈时不留层宽。
2. **前向比反向更早进入通信瓶颈**：阈值 $1+BW/C$ < $1+2BW/C$。直觉：前向通信量与反向一样，但前向计算量只有反向一半。
3. **与 DDP 对比**：
   - DDP 反向：$N_{\mathrm{DP}} \ge 1 + BW/C$（all-reduce 带因子 2，但只发生在反向）；
   - FSDP 前向：$N_{\mathrm{FSDP}} \ge 1 + BW/C$（与 DDP 反向**同形**，因为 FSDP 前向多了 all-gather）；
   - FSDP 反向：$N_{\mathrm{FSDP}} \ge 1 + 2BW/C$（通信减半段、计算加倍，阈值翻倍）。

等价写法（「每台要分到多少样本才养得起通信」）：

$$
\frac{B}{N_{\mathrm{FSDP}}} \gtrsim \frac{C}{W} \quad \text{（前向）}, \qquad
\frac{B}{N_{\mathrm{FSDP}}} \gtrsim 2\cdot\frac{C}{W} \quad \text{（反向）}.
$$

---

## 5. 数值例：XL + RTX 5090（与 DP 报告同一套数）

$d_{\mathrm{model}}=2560$，$d_{\mathrm{ff}}=10240$，$C\approx 2\times 10^{14}$ FLOP/s，$W\approx 5\times 10^{10}$ 字节/s（PCIe 5.0 x16），$C/W = 4000$ FLOP/字节。

| 量 | 值 |
|----|-----|
| $S = 6DD_{\mathrm{FF}}$ | $\approx 157\ \mathrm{MB}$ |
| 前向临界点 $1 + BW/C$ | 见下表 |
| 反向临界点 $1 + 2BW/C$ | 见下表 |

| $B$ | 前向：$1+BW/C$ | 反向：$1+2BW/C$ |
|----:|----------------:|----------------:|
| 64 | 1.02 | 1.04 |
| 4000 | 2.00 | 3.00 |
| 8000 | 3.00 | 5.00 |

取 $B=64$，$N_{\mathrm{FSDP}}=4$：

```
T_compute,fwd ≈ 6·64·2560·10240 / (4 · 2×10¹⁴)  ≈ 12.5 μs
T_comm,fwd    ≈ (3/4) · 157 MB / (50 GB/s)       ≈ 2.4 ms

T_compute,bwd ≈ 25 μs
T_comm,bwd    ≈ 2.4 ms
```

前向、反向都是通信慢两个数量级以上——**从第 2 台 FSDP 起，前向 all-gather 就已经是瓶颈**；这与 DDP「反向 all-reduce 瓶颈」类似，只是 FSDP 把痛点提前到了前向。

真实训练里 FSDP 仍能工作，靠与 DDP 类似的手段：大有效 batch（梯度累积）、BF16 减半通信、层间 **预取 all-gather** 与计算重叠（见 [distributed-communication.md §9](./distributed-communication.md)）。这些不改变上式给出的**裸瓶颈点**，但能把 wall-clock 里的有效通信藏起来。

---

## 6. DDP 与 FSDP 公式汇总

符号：$S = 6\,D\,D_{\mathrm{FF}}$（本层三个权重矩阵的 FP16 总字节数）；DDP 设备数 $N_{\mathrm{DP}}$；FSDP 设备数 $N_{\mathrm{FSDP}}$；环形 collective。

| | DDP | FSDP |
|--|-----|------|
| **前向 FLOPs**（每设备） | $\dfrac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{DP}}}$ | $\dfrac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$ |
| **反向 FLOPs**（每设备） | $\dfrac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{DP}}}$ | $\dfrac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}}$ |
| **前向计算时间** $T_{\mathrm{compute,fwd}}$ | $\dfrac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{DP}}\,C}$ | $\dfrac{6\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}\,C}$ |
| **反向计算时间** $T_{\mathrm{compute,bwd}}$ | $\dfrac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{DP}}\,C}$ | $\dfrac{12\,B\,D\,D_{\mathrm{FF}}}{N_{\mathrm{FSDP}}\,C}$ |
| **前向通信时间** $T_{\mathrm{comm,fwd}}$ | $0$ | $\dfrac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}} \cdot \dfrac{S}{W}$ |
| **反向通信时间** $T_{\mathrm{comm,bwd}}$ | $\dfrac{2\,(N_{\mathrm{DP}}-1)}{N_{\mathrm{DP}}} \cdot \dfrac{S}{W}$ | $\dfrac{N_{\mathrm{FSDP}}-1}{N_{\mathrm{FSDP}}} \cdot \dfrac{S}{W}$ |
| **前向通信瓶颈** | — | $N_{\mathrm{FSDP}} \ge 1 + B \cdot \dfrac{W}{C}$ |
| **反向通信瓶颈** | $N_{\mathrm{DP}} \ge 1 + B \cdot \dfrac{W}{C}$ | $N_{\mathrm{FSDP}} \ge 1 + 2B \cdot \dfrac{W}{C}$ |
