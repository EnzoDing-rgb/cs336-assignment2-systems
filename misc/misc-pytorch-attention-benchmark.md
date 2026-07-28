# PyTorch Attention Benchmarking：认知地图

> Problem **pytorch_attention (a)**。在 GPU 上对 **scaled dot-product attention（缩放点积注意力，下文简称 SDPA）** 做规模扫描：**算力（时间）** 与 **显存** 都要。

---

## 1. 这个实验到底在干嘛？

题面同时要求三件事，不是二选一：

| 测什么 | 题面要求 | 为什么要测 |
|--------|----------|------------|
| **前向时间** | 100 次 forward，每次后 sync | 朴素实现算力随 S 按 **S²** 涨（QKᵀ 与 注意力权重×V 都是序列平方级）；后面 FlashAttention 要证明 **又快又省显存** |
| **反向开始前显存** | backward 前读 `memory_allocated` | 看清为反向保存的 **S×S 矩阵** 占多少；这是 OOM 主因，也是 FlashAttention 要消掉的东西 |
| **反向时间** | 100 次 backward，每次后 sync | 反向同样依赖已物化的 S×S；序列一长，**算力与显存一起爆** |

一句话：**在「没有 Flash、没有多头」的最简设定下，画出朴素 attention 随 (d, S) 变贵、变胖、何时 OOM 的底图**；后题用融合实现替换这条曲线。

本题 **不涉及 optimizer**：没有可学习参数，只有 Q/K/V 张量上的前向与反向。

---

## 2. 实验设定

- **batch B = 8**，**单头**（无 head 维）：Q、K、V 均为 **(B, S, d)**  
- **d** ∈ {16, 32, 64, 128}（题面称 d_model，此处即 embedding 维）  
- **S** ∈ {256, 1024, 4096, 8192, 16384}（题面网格）；若在 A800 80GB 上全程不 OOM，可 **加长 S** 使 OOM 出现在网格后段（见 §5）  
- 实现：`cs336_basics.model.scaled_dot_product_attention`  
- 无 causal mask；FP32；硬件：NVIDIA A800-SXM4-80GB  

---

## 3. 一轮实验量什么？（forward + backward 成对）

**一轮 = 一次 forward + 一次 backward**（中间 `cuda.synchronize`），这才是一个完整训练步在 attention 算子上的缩影。

推荐流程（每格配置）：

```
warmup：若干轮 forward → backward → zero_grad
计时阶段：重复 100 轮
  forward → sync → 累计前向时间
  （首轮 forward 结束后可读「backward 前显存」）
  backward → sync → 累计反向时间
  zero_grad
报告：前向均值 ms、反向均值 ms、backward 前显存（及可选 forward 段 max_memory）
```

题面写的「100 次 forward」「100 次 backward」用 **100 轮成对循环** 满足：每轮只留 **一张** 计算图，避免 100 次 forward 叠图导致过早 OOM。

---

## 4. 显存账本（Assignment 1 口径）

单头、形状 (B, S, d)，为反向典型保存：

| 项 | 形状 | 随 S |
|----|------|------|
| attention scores（softmax 前） | (B, S, S) | S² |
| attention weights（softmax 后） | (B, S, S) | S² |
| Q、K、V | 各 (B, S, d) | 线性 |

主导：**2 · B · S² · 4** 字节（两张 S×S，FP32）。  
S 翻倍 → 保存项约 **×4**。消除思路：**FlashAttention**——不物化完整 S×S。

---

## 5. A800 80GB 与 OOM 边界

题面 4×5=20 格，在 **实现正确、每轮 forward+backward** 前提下，孤立 attention 峰值粗算常 **远低于 80GB**（S=16384 时两张 S×S 约 16GiB 量级），**可能 20 格全部成功**。

若需 OOM 落在网格 **后 1/4～后几格**（而非第 3～5 格）：

- **勿**叠 100 张 forward 图（会在小 S 就假 OOM）  
- **可**在题面 S 列表后追加更大 S，例如 24576、32768（总格数 24～28），使最大几格在 80GB 上触顶  

粗算：每张 S×S 约 `8·S²·4` 字节；两张合计 `64·S²` 字节。S=32768 → 单张约 32GiB，两张约 64GiB，加反向临时与 QKV 后接近或超过 80GB——OOM 会出现在 **最大 S** 附近，符合「后段才 OOM」。

**耗时粗估（100 轮/格，含 warmup）：**

| 规模 | 单格量级 |
|------|----------|
| S≤4096 | 数秒～十几秒 |
| S=8192～16384 | 十几秒～两分钟 |
| S≥24576（若加） | 可达数分钟；OOM 格更快失败 |

**20 格全成功：** 约 **10～25 分钟**  
**28 格（加 24576、32768）：** 约 **20～45 分钟**（视大 S 是否 OOM 中断）

---

## 6. 交付物

| 产出 | 说明 |
|------|------|
| `cs336_systems/pytorch_attention/` | benchmark + sweep + 出图 |
| `reports/pytorch-attention.md` | 完整中文叙述；图嵌正文；**写全 scaled dot-product attention，缩写 (SDPA) 仅作括注** |
| 图 | 前向/反向时间 vs S；显存 vs S；OOM 标记；可选热力图 |

题面英文问点：各配置 timing 或 OOM、OOM 从哪格起、最小 OOM 配置显存账本、backward 保存随 S 如何变、如何消除 → 报告一条线全覆盖，可略超题面但不散。

---

## 7. 与已有实验的关系

| 已有 | 本问 |
|------|------|
| memory profiling 整网 xl | 孤立 attention 算子 |
| measure_block_saved_tensors | 单层 block 内多项；此处只 isolates SDPA |
| FlashAttention（待实现） | 本实验是朴素基线 |
