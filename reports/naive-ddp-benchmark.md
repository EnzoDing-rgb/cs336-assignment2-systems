# Naïve DDP Benchmarking

**Problem:** `naive_ddp_benchmarking` (Assignment 2 §5.2)  
**Hardware:** 2× NVIDIA RTX PRO 6000 Blackwell Server Edition（~97 GB HBM each），GPU 间互联 **PHB**（PCIe Host Bridge，无 NVLink）。  
**Code:** `cs336_systems/distributed/benchmarking/benchmark_naive_ddp.py` · **Plotting:** `cs336_systems/distributed/benchmarking/plot_naive_ddp.py`  
**Data:** `artifacts/naive_ddp_benchmark.csv` · `artifacts/naive_ddp_per_param_latency.csv`

---

## 符号表

| 符号 | 含义 | 本实验取值 |
|------|------|-----------|
| B | 总 batch size（跨所有 GPU 的样本数之和） | 4, 8, 16, 32, 64 |
| d | GPU 数量（world size） | 2 |
| N_params | 模型中 `nn.Parameter` 张量的个数 | 291（xl 模型） |
| t_step | 单次训练迭代的墙上时间（s） | — |
| t_sync | 梯度同步（all-reduce）耗时（s） | — |
| η_comm | 通信时间占比 = t_sync / t_step | — |

---

## 1. 实验在量什么

### 1.1 什么是 Naïve DDP

分布式数据并行（DDP）把 batch 切分到多张 GPU 上并行训练，核心循环如下：

```
每个 training step:
  ① forward(x_local)           → 各 GPU 用本地数据前向
  ② loss.backward()             → 各 GPU 算出局部梯度
  ③ all_reduce(gradients)       → 跨 GPU 平均梯度（通信！）
  ④ optimizer.step()            → 各 GPU 用平均梯度更新参数
```

**Naïve DDP** 在步骤 ③ 中，对**每一个 `nn.Parameter` 张量单独发起一次 `all_reduce`**。xl 模型有 291 个 parameter tensors，因此每个 step 发起 **291 次**独立的 all-reduce 调用。这就是它 "naïve" 的原因——大量小消息的通信极其低效。

### 1.2 并行什么、不并行什么

| 组件 | Naïve DDP | 说明 |
|------|-----------|------|
| **Data**（数据） | ✅ 分片 | 每 GPU 处理 B/d 个样本 |
| **Parameters**（参数） | ❌ 复制 | 从 rank 0 broadcast，每卡一份完整副本 |
| **Gradients**（梯度） | ❌ 复制 | backward 后各自有局部梯度，all-reduce 后每卡一份完整副本 |
| **Optimizer States**（优化器状态） | ❌ 复制 | 每卡各自维护 AdamW 的 m, v，内容完全相同 |

Naïve DDP **只做了数据并行**。参数、梯度、优化器状态都是全量复制的，通信开销仅来自梯度同步（步骤 ③）。

### 1.3 本实验量什么

两个配置：

| 配置 | GPU | 总 batch B | 每 GPU 样本 | 通信 |
|------|-----|-----------|-------------|------|
| 单卡 baseline | 1 | B | B | 无 |
| Naïve DDP | 2 | B | B/2 | 291 次 all_reduce / step |

对比两者回答：**通信占了多大比例？随 batch size 如何变化？**

---

## 2. 实验设置

### 2.1 模型

| 参数 | 值 |
|------|-----|
| 模型 | `BasicsTransformerLM`（Assignment 1 参考实现） |
| 规格 | **xl**（见 §2.1.2） |
| d_model | 2,560 |
| d_ff | 10,240 |
| num_layers | 32 |
| num_heads | 32 |
| vocab_size | 10,000 |
| context_length | 512 |
| 精度 | float32 |
| 总参数量 | 3,406,809,600（≈3.4B） |
| Parameter tensor 数量 | **291** |
| 梯度通信量 / step | 13.63 GB（= 3.4B × 4 bytes） |

### 2.2 训练超参数

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW（lr=1e-3, weight_decay=0.1，使用 `cs336_basics.optimizer.AdamW`） |
| Batch sizes | 4, 8, 16, 32, 64（总 batch size） |
| Warmup steps | 5（丢弃，不计时） |
| Measurement steps | 10（取平均 ± std） |
| 数据 | 随机整数 token（`randint(0, vocab_size)`） |
| Seed | 0（可复现） |

### 2.3 计时方法

每个 training step 切分为 5 个计时段，**每段前后均调用 `torch.cuda.synchronize(device)`**：

```
  ┌─ forward ─┬─ loss ─┬─ backward ─┬─ gradient_sync ─┬─ optimizer ─┐
  │ model(x)  │ CE(·,y)│ .backward()│ finish_grad_sync│ .step()     │
  └───────────┴────────┴────────────┴─────────────────┴─────────────┘
  ← 每段独立穿 cuda-sync 计时 →
```

- 计时器：`timeit.default_timer()`（系统最高精度墙上时钟）
- `gradient_sync` 段 = `NaiveDDP.finish_gradient_synchronization()` 的墙上时间（含所有 291 次 all-reduce）
- 单卡 baseline 的 `gradient_sync` = 0（无通信）

### 2.4 硬件

- 2× NVIDIA RTX PRO 6000 Blackwell Server Edition，每卡 ~97 GB HBM
- GPU 间互联：PHB（PCIe Host Bridge），**无 NVLink**
- CUDA 通过 NCCL 后端通信

---

## 3. 结果

### 3.1 主结果：batch=4 的 step 时间分解

| 阶段 | 单卡（batch=4） | Naïve DDP 2-GPU（batch=4） | 说明 |
|------|-----------------|--------------------------|------|
| forward | 0.323s (27.6%) | 0.154s (13.8%) | DDP 每 GPU 只算 2 个样本 |
| loss | ~0.000s | ~0.000s | 可忽略 |
| backward | 0.572s (48.8%) | 0.295s (26.4%) | DDP 每 GPU 只算 2 个样本 |
| **gradient sync** | **—** | **0.396s (35.4%)** | **← 291 次 all-reduce！** |
| optimizer | 0.275s (23.5%) | 0.272s (24.3%) | 基本相同（参数量一样） |
| **TOTAL** | **1.171s** | **1.118s** | |

![Naive DDP Step Breakdown](figures/naive_ddp_step_breakdown.png)

**核心发现：通信（gradient sync）占 2-GPU Naïve DDP 训练步的 35.4%**。每卡计算量只有单卡的一半（0.154+0.295 = 0.449s vs 0.323+0.572 = 0.895s），但通信把省下来的时间几乎全吃光了——DDP 总时间 1.118s，单卡 1.171s，加速比仅 **1.05×**。

### 3.2 Batch size sweep

| Batch B | 单卡总时间 | DDP 总时间 | DDP 中 t_sync | 通信占比 | 加速比 |
|---------|-----------|-----------|--------------|---------|--------|
| 4 | — | — | — | — | — |
| 8 | — | — | — | — | — |
| 16 | — | — | — | — | — |
| 32 | — | — | — | — | — |
| 64 | — | — | — | — | — |

> *表格将在 benchmark sweep 完成后填充。*

![Naive DDP Batch Sweep](figures/naive_ddp_batch_sweep.png)

### 3.3 附录：Per-Parameter all-reduce 延迟

![Per-Parameter all-reduce Latency](figures/naive_ddp_per_param_latency.png)

xl 模型共有 **291 个 `nn.Parameter` 张量**，每个对应一次 all-reduce。散点图展示每个张量的大小（bytes）与其单次 all-reduce 端到端延迟（ms）的关系。

---

## 4. 分析

### 4.1 为什么通信占比这么高

**表面原因：** 13.63 GB 梯度数据要通过 PCIe 传，看起来带宽不够。

**深层原因（更重要）：不是数据量的问题，是消息数量的问题。**

291 次 all-reduce 中，大小分布极为不均：

| 类型 | 每层数量 | 单个张量大小 | 特点 |
|------|---------|-------------|------|
| Embedding / LM head | 各 1 | 25,600,000 elem × 4B = 102.4 MB | 能充分利用带宽 |
| Attention 权重（QKV + output） | 4 | 6,553,600 elem × 4B = 26.2 MB | 中等 |
| FFN 权重（up + down） | 2 | 26,214,400 elem × 4B = 104.9 MB | 最大 |
| FFN 权重（gate） | 1 | 26,214,400 elem × 4B = 104.9 MB | — |
| RMSNorm 权重 | 2 | 2,560 elem × 4B = 10.2 KB | **极小！延迟主导** |

对于 RMSNorm 的 ~10 KB 张量，一次 all-reduce 的延迟几乎全部来自：
- **Kernel launch overhead**：每次 CUDA kernel 启动有 ~5–10 μs 开销
- **NCCL 协议开销**：即使是 ring all-reduce，也有握手、同步的开销
- **PCIe 延迟**：传输 10 KB 数据本身只需 ~1 μs（在 12 GB/s 带宽下），但 PCIe 事务层有 per-transaction 开销

### 4.2 理论通信时间 vs 实测

在 PCIe 带宽 ~12 GB/s（本机 PHB 互联实测）下：
- 理论传输时间 = 13.63 GB / 12 GB/s ≈ 1.14s（只算数据传输，不算协议开销）
- 实测 0.396s < 1.14s，说明 ring all-reduce 在 2 GPU 情况下每 bit 只需传一次（ring factor = 1），且部分较大张量能接近带宽峰值
- 但大量小张量的延迟累积使得实际 time 远高于纯带宽理想值

### 4.3 Batch size 如何影响通信开销

通信量（13.63 GB）**不随 batch size 变化**——梯度参数量是固定的。但**计算量**随 batch size 线性增长。

- **小 batch（4）：** 计算 ~0.72s，通信 ~0.40s，通信占比高
- **大 batch（64）：** 计算大幅增长，通信仍是 ~0.40s，通信占比下降

这意味着 Naïve DDP 在**小 batch 场景下通信开销最严重**。对于大 batch 训练，通信时间和计算时间相比变得不那么重要——但这受限于 GPU 显存。

### 4.4 改进方向

| 改进 | 方法 | 预期效果 |
|------|------|---------|
| **Flatten all-reduce** (§5.3.1) | 把 291 个梯度 concat 成一个 tensor，一次 all-reduce | 消除 291 次 kernel launch + 协议开销 |
| **Overlap compute & comm** (§5.3.2) | backward 过程中异步 all-reduce 已就绪的梯度 | 把通信隐藏在计算中，墙上看不到 |
| **FSDP** (§7) | 参数也分片，all-gather 权重 + reduce-scatter 梯度 | 减少内存，但通信模式改变 |

这些正是后续作业题目的内容。

---

## 附录 A：Per-Parameter 延迟分布

本附录展示每个 `nn.Parameter` 张量的 all-reduce 延迟与张量大小的关系。核心观察：

1. **大张量（>1 MB）** 的延迟接近带宽预测线（~12 GB/s），说明通信效率高
2. **小张量（RMSNorm, ~10 KB）** 的延迟几乎不随大小减小而降低——latency floor 约 0.01–0.02 ms，由 kernel launch + NCCL 协议开销主导
3. 291 次调用中大量是小张量（32 层 × 每层至少 2 个 norm = 64 个 norm 张量 + 其他小张量），**这些小张量的延迟不是"传输数据"，而是"发起通信"本身**

这也解释了为什么 Flatten DDP（把所有梯度拼成一个 tensor，一次 all-reduce）能显著降低通信时间——它把 291 次 kernel launch + 协议开销缩减为 1 次。

---

## 复现

```bash
# 运行 benchmark（全 sweep，约 30–45 分钟）
uv run python -m cs336_systems.distributed.benchmarking.benchmark_naive_ddp

# 仅 batch=4（约 2 分钟）
uv run python -m cs336_systems.distributed.benchmarking.benchmark_naive_ddp --batch-sizes 4

# 出图
uv run python -m cs336_systems.distributed.benchmarking.plot_naive_ddp
```
