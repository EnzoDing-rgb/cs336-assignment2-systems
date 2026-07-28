# Torch Compile 作业题：认知地图

> 对应 Problem **torch_compile** (a)(b)，讲义 §4.2 *Benchmarking JIT-Compiled Attention*。  
> 上一题（scaled dot-product attention 基准）量的是 **朴素 PyTorch 实现** 的时间与显存；本题问：**加上 `torch.compile` 之后，同样的代码快多少？**

---

## 0. 题面在说什么

PyTorch 2.0 起自带 **即时编译器（just-in-time compiler，JIT）**：`torch.compile` 在运行时分析计算图，尝试生成融合的 Triton/CUDA 内核，减少 kernel 启动次数与中间张量读写。

接口极简：

```python
layer = SomePyTorchModule(...)
compiled_layer = torch.compile(layer)
# compiled_layer 在语义上与 layer 等价（前向、反向行为一致）
```

也可以 `torch.compile(model)` 编译整网，或 `torch.compile(python_fn)` 编译一个调用 PyTorch 算子的 Python 函数。

本题分两步：

| 部分 | 编译对象 | 对比什么 | 交付物 |
|------|----------|----------|--------|
| **(a)** | 缩放点积注意力（scaled dot-product attention，SDPA）实现 | 与上一题 **相同配置** 下的未编译版 | 前向 / 反向时间对照表 |
| **(b)** | 整网 `BasicsTransformerLM` | 与现有 end-to-end 基准里的未编译版 | 前向、前向+反向+optimizer 时间对照表 |

---

## 1. `torch.compile` 到底做了什么（第一性原理）

朴素 PyTorch 执行路径：**Python 解释器 → 逐个 dispatch 算子 → 每个算子一次 GPU kernel**。  
算子之间若有可融合的模式（例如连续 elementwise、或 RMSNorm 里的 reduce + scale），仍会 **多次读写 HBM**。

`torch.compile` 的典型路径（简化）：

```
Python 函数 / nn.Module
        ↓  TorchDynamo 捕获计算图
   FX Graph / Inductor
        ↓  融合、调度
   Triton / CUDA 内核（可能少次、大块）
```

对本作业有意义的几点：

1. **语义不变**：输出应与 eager 模式一致（在数值误差允许范围内）；梯度也应可反传。
2. **首次调用很慢**：第一次 forward 要做 **图捕获 + 编译**，可能秒级甚至更长；**计时时必须 warmup**，不能把编译时间算进 steady-state。
3. **可能融合 attention 周围的 einsum + softmax**，但 **不保证** 变成 FlashAttention；仍可能物化 S×S，显存主导项未必大幅下降。
4. **主要收益往往在算力（时间）**：kernel 融合、减少 launch overhead；显存节省是副产品，不是本题重点。

与讲义 §1.1（RMSNorm + `torch.compile` + `saved_tensors_hooks`）的关系：那边用 **单算子** 说明融合如何 **减少保存张量**；本题 (a) 用 **同一套 attention 网格** 看 **端到端算子时间** 是否下降。

---

## 2. Part (a)：编译版 scaled dot-product attention

### 2.1 目标

在 **与 pytorch_attention (a) 完全相同的配置** 下，增加一列 **compiled**，交付 **未编译 vs 编译** 的前向、反向时间对照表。

### 2.2 配置（与已有基准对齐）

沿用 `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/attention_operator/` 的设定：

| 项 | 值 |
|----|-----|
| 实现 | `cs336_basics.model.scaled_dot_product_attention` |
| batch B | 8 |
| 单头 | Q、K、V 形状 `(B, S, d)` |
| d | 16, 32, 64, 128 |
| S | 256 … 32768（与已跑网格一致，含扩展格） |
| 精度 | FP32 |
| 每格 | 100 轮 forward → backward，每段后 `cuda.synchronize` |

题面 (a) **只要求时间对照表**，不要求重复显存扫描（上一题已做）；若实现顺手可记峰值，但 **交付物是表**。

### 2.3 怎么包 `torch.compile`

两种等价思路（选一种即可）：

**方案 A：编译 Python 函数（推荐，改动小）**

```python
from cs336_basics.model import scaled_dot_product_attention

def attention_fn(Q, K, V):
    return scaled_dot_product_attention(Q=Q, K=K, V=V, mask=None)

compiled_attention = torch.compile(attention_fn)
```

**方案 B：编译 `nn.Module` 包装**

```python
class AttentionOp(nn.Module):
    def forward(self, Q, K, V):
        return scaled_dot_product_attention(Q, K, V, mask=None)

compiled_attention = torch.compile(AttentionOp())
```

计时循环里：**eager 路径** 直接调原函数；**compiled 路径** 调 `compiled_attention(...)`。Q、K、V 创建方式与 `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/attention_operator/benchmark.py` 相同。

### 2.4 Warmup 与计时（极易踩坑）

`torch.compile` 的 **第一次** 调用包含编译；若 warmup 不足，会把编译算进均值。

推荐每格配置：

```
1. 新建 compiled 函数（或每格重新 compile，避免 dynamo 缓存混淆 — 通常同一形状重复 compile 一次即可）
2. warmup：≥10 轮 forward → backward（大 S 时可酌情减少轮数，但编译必须完成）
3. 正式计时：100 轮 forward → backward（与上一题相同）
4. 每轮 forward / backward 后 cuda.synchronize
```

**表头建议：**

| d | S | eager forward (ms) | compiled forward (ms) | eager backward (ms) | compiled backward (ms) | 前向加速比 | 反向加速比 |
|--:|--:|-------------------:|----------------------:|--------------------:|-----------------------:|-----------:|-----------:|

加速比 = eager 时间 ÷ compiled 时间（>1 表示编译版更快）。

可对 **代表性几格**（如 S=4096、16384）全表展示；或 **全网格** 一张大表 — 题面未限行数，全表更利于写报告。

### 2.5 预期现象（复习用）

| S 规模 | 预期 |
|--------|------|
| 小 S | 算子本身太快，编译收益可能被 launch 噪声淹没，加速比接近 1 |
| 中 S | einsum + softmax 融合可能有 **可见加速**（例如 1.2×–2×，视 PyTorch 版本与 GPU） |
| 大 S | 仍受 S² 算力与显存限制；compile **不会** 把 OOM 格 magically 变可跑（除非融合显著减中间张量，对朴素 attention 不保证） |

**(a) 不替代 FlashAttention**：compile 是编译器自动融合；FlashAttention 是算法级 IO 优化。后面还会有专门对比。

### 2.6 代码落点（规划）

| 用途 | 路径 |
|------|------|
| 统一入口 | `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/torch_compile/suite.py`（`--part attention`） |
| 扩展基准 | `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/attention_operator/benchmark.py`（`use_compile` 开关） |
| 数据 | `/root/.dev/ml-sys/cs336/assignment2-systems/artifacts/attention_operator/compile_results.json` |
| 报告 | `/root/.dev/ml-sys/cs336/assignment2-systems/reports/benchmarking-torch-compile.md` |

与上一题关系：**同一网格、同一计时协议**，多一列 compiled。

---

## 3. Part (b)：编译整网 Transformer

### 3.1 目标

在 **end-to-end 基准脚本** 里对 **整个 `BasicsTransformerLM`** 做 `torch.compile(model)`，与 vanilla 对比：

- **前向** 时间怎么变？
- **前向 + 反向 + optimizer step**（完整训练步）怎么变？

交付物：**vanilla vs compiled 对照表**。

### 3.2 现有基准在哪里

| 项 | 路径 |
|----|------|
| 脚本 | `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/e2e_timing/e2e.py` |
| 入口 | `uv run --no-sync python -m cs336_systems.e2e_timing` |
| 已有报告 | `/root/.dev/ml-sys/cs336/assignment2-systems/reports/end2end-benchmark.md` |
| 模式 | `timed_train`：分段计时 forward / loss / backward / optimizer |

当前设定：vocab=10000，batch=4，context=512，AdamW，FP32，warmup=5，steps=10；model size = small / medium / large / xl（10b 在 80GB 上 OOM）。

### 3.3 怎么改

在 `BenchmarkConfig`（或等价配置）增加开关，例如 `use_compile: bool`：

```python
model = BasicsTransformerLM(...)
if cfg.use_compile:
    model = torch.compile(model)
```

**整网 compile 的 warmup 比 (a) 更重**：第一次 `timed_train` 步可能包含 **整图编译**（所有 layer、attention、FFN）。建议：

- warmup ≥ 5（与现网一致），必要时 **compiled 单独多加几步** 直到时间稳定；
- 同一 size **先跑 vanilla 再跑 compiled**（或分两次进程），避免 compile 污染 allocator 状态难以解释；
- 记录 **编译是否成功**（个别 op 可能 graph break，仍能用但加速有限）。

### 3.4 表怎么设计

**表 1：前向（仅 forward 段或 `mode=forward`）**

| model size | vanilla forward (ms) | compiled forward (ms) | 加速比 |
|------------|---------------------:|----------------------:|-------:|
| small | … | … | … |
| … | | | |

**表 2：完整训练步（forward + loss + backward + optimizer）**

| model size | vanilla full step (ms) | compiled full step (ms) | 加速比 |
|------------|----------------------:|------------------------:|-------:|
| small | … | … | … |
| … | | | |

可与现有 `end2end-benchmark.md` 的 **分段表** 并列：compiled 行是否 **backward / optimizer 也加速**，是报告里要写的观察点（attention + FFN 融合主要落在 forward/backward 算子段）。

### 3.5 预期现象

| 观察 | 原因 |
|------|------|
| 首次 step 极慢 | 整网编译 |
| steady-state 前向加速 | 多层算子融合、减少 Python 开销 |
| backward 也可能加速 | Inductor 生成融合反向内核 |
| optimizer 段变化小 | AdamW 多为 elementwise，已较快 |
| xl 收益可能更明显 | 绝对时间长，固定开销占比低 |

显存本题 **未要求** 对比；若 compiled 减少中间张量，peak 可能略降，可作附加说明。

### 3.6 代码落点（规划）

| 用途 | 路径 |
|------|------|
| 统一入口 | `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/torch_compile/suite.py` |
| 扩展 e2e | `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/e2e_timing/e2e.py`（`use_compile` 开关） |
| 扩展 attention | `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/attention_operator/benchmark.py`（`use_compile` 开关） |
| 数据 | attention: `artifacts/attention_operator/compile_results.json`；e2e: `artifacts/e2e_benchmark/compile_suite/manifest.json` |
| 报告 | `reports/benchmarking-torch-compile.md` |

运行：

```bash
python -m cs336_systems.torch_compile              # (a) + (b)
python -m cs336_systems.torch_compile --part attention
python -m cs336_systems.torch_compile --part e2e
python -m cs336_systems.torch_compile --skip-run   # 仅从 JSON 重画图/报告
```

---

## 4. (a) 与 (b) 的关系

```
(a) 孤立 attention 算子 + compile     →  看清「单算子」编译收益，网格与 SDPA 基准相同
(b) 整网 BasicsTransformerLM + compile →  看清「真实训练步」编译收益，含多层 attention + FFN + optim
```

(a) 是 **受控实验**（单算子、无参数地板）；(b) 是 **系统级实验**（参数 + Adam + 32 层叠加）。  
(a) 里看到的加速比，不应直接等同于 (b) 里 full step 的加速比——整网还有 graph break、编译范围、内存带宽等额外因素。

---

## 5. 与仓库里已有实验的对照

| 已有 | 本题 |
|------|------|
| `/root/.dev/ml-sys/cs336/assignment2-systems/reports/benchmarking-scaled-dot-product-attention.md` | (a) 的 **eager 基线**；本题加 compiled 列 |
| `/root/.dev/ml-sys/cs336/assignment2-systems/misc/misc-operator-fusion-rmsnorm.md` | 单算子融合机理（saved tensors） |
| `/root/.dev/ml-sys/cs336/assignment2-systems/scripts/measure_block_saved_tensors.py` | 已用 `torch.compile(block, fullgraph=True)` 测单层融合 |
| `/root/.dev/ml-sys/cs336/assignment2-systems/reports/end2end-benchmark.md` | (b) 的 **vanilla 基线** |
| `misc/misc-flash-attention-triton-intro.md` | 下一题 Flash Attention：Triton 入门（讲义 §4.2.1 加权求和） |

---

## 6. 实施顺序建议

1. **(a)** `python -m cs336_systems.torch_compile --part attention` → 扫同一网格 → 出 eager vs compiled 时间表。
2. **(b)** `python -m cs336_systems.torch_compile --part e2e` → 对 small→xl 跑 vanilla / compiled → 出 forward 与 full step 表。
3. 写报告：每张表配 **2–3 句** 解释（小 S 噪声、大模型收益、编译 warmup 已剔除）。

粗估耗时（A800）：

| 部分 | 量级 |
|------|------|
| (a) 全网格 ×2（eager 已有可 skip-run） | 约 10–50 分钟（compiled 首格编译慢） |
| (b) 4 sizes ×2 | 约 15–30 分钟 |

---
