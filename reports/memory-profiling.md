# Memory Profiling

**设定：** `BasicsTransformerLM` **xl**；batch=4；context ∈ {128, 512}；mode ∈ {forward, train}；precision ∈ {FP32, BF16}；warmup=2；train = forward + loss + backward + `AdamW.step()`。

**关于 handout 的 128/2048（为何不用 2048）：**

讲义默认 batch $B=4$，并要求 xl 在 context $S\in\{128,2048\}$ 上做完整训练步；后续 activation checkpointing 示例也写死了形状 `[4, 2048, 2560]`，即 $B=4,\ S=2048,\ d_{\mathrm{model}}=2560$。

先解释 attention 中那块随 $S^2$ 增长的显存从何而来。在每一层中，token 首先被投影为 $Q,K,V$（每个 head 的维度为 $d_k=d_{\mathrm{model}}/H$），然后计算

$$
\mathrm{scores}=QK^{\top}/\sqrt{d_k}\in\mathbb{R}^{B\times H\times S\times S},
$$

再经 `softmax` 得到注意力权重，最后乘以 $V$。为了支持反向传播，autograd 通常需要将这份 **$S\times S$ 的 score（以及/或 softmax 后的权重）逐层保存下来**。xl 的超参为 $H=32$ 头、$L=32$ 层；FP32 每元素占 4 bytes。若每层只计 **一张** score 图、且 32 层在反向传播前同时存活，则

$$
\underbrace{B}_{\text{batch}}\times\underbrace{H}_{\text{heads}}\times\underbrace{S\cdot S}_{S\times S\text{ 矩阵}}\times\underbrace{4}_{\text{bytes/elem}}\times\underbrace{L}_{\text{层数}}
=4\cdot 32\cdot 2048^{2}\cdot 4\cdot 32
=2^{36}\ \text{bytes}=64\ \text{GiB}.
$$

这里 64 GiB **并非**「整次训练只需 64 GiB」。它仅是「所有层的 attention score 缓存」这一项的下界；同一次前向中还有残差流激活 $(B,S,d)$、FFN 中间激活、往往还有第二份 $S\times S$（softmax 输出）、权重本身（约十几 GiB），完整训练步还要再加 AdamW 的两份状态（约为参数量的两倍）。因此即便使用 80 GiB 的卡，「64 GiB 的 score 加上其它一切」也会明显超过 80 GiB。实证上，同设定 $B=4$ 在 $S=1024$ 时前向已 OOM（见 `/root/.dev/ml-sys/cs336/assignment2-systems/reports/nsys-profile.md`）。

因此本报告改用 **$S\in\{128,512\}$、仍保持 $B=4$**：与全文其它实验使用同一 batch，(b) 的横向对比只改变 context，避免「为塞进 2048 而改 batch」把变量搅在一起。

**交付范围：** (a)–(e) 来自 8 格 PyTorch memory snapshot 套件；(f) 来自 headless Nsight（`--cuda-memory-usage` + TransformerBlock NVTX）。代码位于 `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/memory_profiling/`。

## (a) 活跃显存时间线

**录制方式：** warmup 后开启 `_record_memory_history`，跑 **一步**，再 `_dump_snapshot`。曲线按 memory_viz 语义重建（`alloc` 为 +，`free_completed` 为 −）。竖虚线是阶段边界的 **地面真值**（在 `cuda.synchronize` 之后打点），并非事后猜测。

**仅前向（ctx=512, FP32）**

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_a_xl_ctx512_forward.png" alt="memory_a_xl_ctx512_forward" width="560" />

**完整训练步（ctx=512, FP32）**

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_a_xl_ctx512_train.png" alt="memory_a_xl_ctx512_train" width="560" />

**训练步各阶段 max_allocated（FP32）**

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_a_staged_peaks_fp32_train.png" alt="memory_a_staged_peaks_fp32_train" width="520" />

**解答 (a):** 仅前向（无权重、无 Adam）时显存爬升至约 **39.750 GiB** 后进入平台期，直到释放激活。完整训练步的基线已包含权重和 AdamW（约 50.908 GiB），前向将激活堆叠上去，在 `forward`/`loss` 附近达到整步峰值 **65.538 GiB**；`backward` 呈台阶式下降（逐层释放为反传而暂存的中间结果），结束后驻留约 **50.985 GiB**；`optimizer` 几乎平坦（约 50.985 GiB），因为 Adam 状态早在 warmup 阶段就已分配完毕。因此：**前向=爬升/高平台，反向=台阶下降，优化器=平坦**——结合图中的阶段竖线，三段可以明确区分。

## (b) 不同 context 下的峰值显存

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_b_peaks_by_context.png" alt="peaks by context" width="560" />

| context | forward peak (GiB) | train peak (GiB) |
|--------:|-------------------:|-----------------:|
| 128 | 18.053 | 51.416 |
| 512 | 39.750 | 65.538 |

单位：`torch.cuda.max_memory_allocated` 在该次被 profile 的 step 上的全局峰值（GiB）。
训练步远大于前向的原因：完整一步除了权重之外，还要常驻 AdamW 状态，并在反向阶段短暂叠加上梯度；
context 变长时两者都会增长，但 attention 的 $S\times S$ 项使涨幅快于线性。

## (c) 混合精度（BF16）

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_c_bf16_vs_fp32.png" alt="BF16 vs FP32 memory" width="640" />

左图为 ctx=512 的实测峰值；右图将训练步峰值拆分为「权重 / Adam（始终为 FP32）」与「激活（仅 autocast 时才可能变窄）」的示意，说明即便激活理想减半，总峰值也远不会减半。

| context | mode | FP32 peak (GiB) | BF16 peak (GiB) | Δ |
|--------:|------|----------------:|----------------:|---|
| 128 | forward | 18.053 | 22.452 | +4.399 GiB (+24.4%) |
| 128 | train | 51.416 | 51.406 | -0.009 GiB (-0.0%) |
| 512 | forward | 39.750 | 36.832 | -2.917 GiB (-7.3%) |
| 512 | train | 65.538 | 62.674 | -2.864 GiB (-4.4%) |

**解答 (c):** `torch.autocast(bf16)` 只将 **部分矩阵乘的激活** 算成 BF16；**参数本身仍为 FP32**，AdamW 的一阶/二阶动量也仍为 FP32。因此峰值显存里「权重 + 优化器状态」这一大块几乎不动——完整训练步的基线就已约 50 GiB（见 (a)），真正可能变窄的只有激活。以 xl·ctx=512 为例：前向峰值 39.750→36.832 GiB（少约 2.9 GiB，−7.3%），训练步峰值 65.538→62.674 GiB（少约 2.9 GiB，−4.4%）；ctx=128 的训练步几乎不变（51.416→51.406 GiB），前向甚至因临时 dtype 转换略增（18.053→22.452 GiB）。换句话说，激活即便理想减半，也只能从总峰值里抠出几个 GiB，绝不可能把 65 GiB 压到 30 GiB 量级。算力侧则不同：BF16 能走 Tensor Core，端到端 step 时间往往能明显缩短（见 `/root/.dev/ml-sys/cs336/assignment2-systems/reports/benchmarking-mixed-precision.md`）；显存峰值这边，对本设定的收益就是「少几个 GiB」，不足以靠混精单独解决 OOM。

## 先把「残差」说清楚（读 (d)(f) 之前）

文中会出现三个容易搅在一起、其实并不相同的概念。按计算图里真实发生的事从上到下排：

**1. 残差连接（skip connection）。**  
这就是何恺明 ResNet 里的 $y=x+f(x)$：旁路把输入原样加到子层输出上，减轻深层网络里梯度难以回传的问题。Transformer 沿用了同一结构。我们的 `BasicsTransformerLM` **确实实现了它**：`TransformerBlock.forward` 里是

```python
attn_sublayer_output = x + x_attn
ffn_sublayer_output = attn_sublayer_output + x_ffn
```

也就是 attention 子层和 FFN 子层各做一次「输入 + 子层输出」。  
这里的 $x$ 在加法里是**引用同一块显存**，并不会因为「传到下一层」就再复制一份——你的直觉在这一步是对的。

**2. 残差流（residual stream）。**  
指在各 `TransformerBlock` 之间传递、并被上述加法不断写回的那条主激活，形状 $(B,S,d_{\mathrm{model}})$。它是「网络主干上正在流动的那条向量」，(d) 问的就是**一张**这样的张量有多大。它和 ResNet 有渊源（加法写回主干），但 (d) 要的只是这个张量的字节数，不是在问「有没有实现 ResNet」。

**3. 为反向传播保存的张量（讲义里的 residuals / saved tensors）。**  
这是另一个词。前向算 $f(x)$ 时，autograd 会把许多**子层内部的中间结果**留下来，反向时才用得上——例如 attention 的 $S\times S$ 分数、FFN 变宽后的中间激活、RMSNorm 的中间量等。讲义 §3 把这些叫 residuals；本文后面统一说 **「保存张量」**，以免和上面的残差连接、残差流混淆。

**为什么显存还会涨、而不能「复用上一层 activation 就完事」？**  
残差连接只保证「主干上的 $x$」可以共享引用；它**不能**省掉 $f(x)$ 内部新造出来的那些中间张量。更关键的是时间顺序：完整训练步是「整网前向先跑完，再整网反向」。在反向开始之前，各层为反传存下的中间结果通常都还活着——第 1 层算完后并不能马上丢掉它的保存张量，因为反向要等到最后一层之后才从后往前走。于是 32 层的保存张量会叠在一起，这才是 (a) 里前向爬升、反向台阶下降的原因，也是 (f) 要按「单层」去量保存量与梯度的原因。  
（若用 activation checkpointing，可以故意不存这些中间结果、反向时再算一遍——那是后文的题，此处不做。）

下面 (d) 量的是概念 2（一张残差流有多大）；(e) 看到的最大块往往来自概念 3 里的 attention 分数；(f) 量的是概念 3（一层为反传存了多少、反向又写出多少梯度）。

## (d) 残差流激活大小（解析推导）

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_d_residual_stream.png" alt="residual stream size" width="640" />

这里的「残差流」即上一节概念 2：主干上形状为 $(B,\,S,\,d_{\mathrm{model}})$ 的主激活——每个 batch、每个 token 位置有一条宽度为 $d_{\mathrm{model}}$ 的向量。xl 的 $d_{\mathrm{model}}=2560$；单精度 FP32 每元素占 4 bytes。因此 **单张** 残差流张量的体积为

$$
\frac{B\cdot S\cdot d_{\mathrm{model}}\cdot 4}{1024^{2}}\ \text{MiB}.
$$

代入本报告设定 $B=4$：

- $S=128$：$4\cdot 128\cdot 2560\cdot 4/1024^{2}=5.00$ MiB
- $S=512$：$4\cdot 512\cdot 2560\cdot 4/1024^{2}=20.00$ MiB
- 讲义参考 $S=2048$：$4\cdot 2048\cdot 2560\cdot 4/1024^{2}=80.00$ MiB

**解答 (d):** 上式即为单精度残差流激活的大小。对本报告设定 $B=4,\ S=512$ 为 **20.00 MiB**；若按讲义 $S=2048$ 则为 **80.00 MiB**。注意这只是「单层接口上的一条数据流」，并非整网峰值显存。

## (e) 最大分配（前向快照）

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_e_attn_score_alloc.png" alt="attention score allocation" width="640" />

从 ctx=512 FP32 前向的快照中，按 `alloc` 体积排序的前 5 大分配（等价于 memory_viz 调低 Detail 后所看到的最大块）：

| rank | size (MiB) | stack (truncated) |
|-----:|-----------:|----------------|
| 1 | 128.0 | functional.py:407 einsum<br>_backends.py:291 einsum<br>einops.py:939 einsum<br>model.py:427 scaled_dot_product_attention |
| 2 | 128.0 | model.py:427 scaled_dot_product_attention<br>model.py:520 forward<br>module.py:1750 _call_impl<br>module.py:1739 _wrapped_call_impl |
| 3 | 128.0 | model.py:430 scaled_dot_product_attention<br>model.py:520 forward<br>module.py:1750 _call_impl<br>module.py:1739 _wrapped_call_impl |
| 4 | 128.0 | nn_utils.py:5 softmax<br>model.py:432 scaled_dot_product_attention<br>model.py:520 forward<br>module.py:1750 _call_impl |
| 5 | 128.0 | nn_utils.py:6 softmax<br>model.py:432 scaled_dot_product_attention<br>model.py:520 forward<br>module.py:1750 _call_impl |

**解答 (e):** 最大的单次分配均为 **128.0 MiB**。其来源与 (d) 不同：这里不是残差流 $(B,S,d)$，而是 attention 中 $QK^{\top}$ 得到的 score（或随后的 softmax 权重），形状为 $(B,H,S,S)=(4,32,512,512)$。体积为

$$
\frac{B\cdot H\cdot S\cdot S\cdot 4}{1024^{2}}
=\frac{4\cdot 32\cdot 512\cdot 512\cdot 4}{1024^{2}}=128\ \text{MiB},
$$

与表中数字一致；调用栈也指向 `scaled_dot_product_attention` 中的 `einsum` / `softmax`。因此「Detail 调低后最大的块」即为 **单层、单份** $S\times S$ attention 激活（在本设定下，每层还会出现多份同类块）。pickle：`/root/.dev/ml-sys/cs336/assignment2-systems/artifacts/memory_profiling/snapshots/xl_ctx512_forward_mpoff/xl_ctx512_forward_mpoff.pickle`（可拖入 [pytorch.org/memory_viz](https://pytorch.org/memory_viz) 复核）。

## (f) 用 Nsight 看「一层」为反向传播存了多少东西

### 1. 要干什么？

前面 (a)–(e) 看的是整网显存曲线和峰值。本题换一个更细的问题，对准的是上一节的**概念 3（保存张量）**，不是概念 1 的残差连接，也不是概念 2 的「一张残差流有多大」：

**只盯住模型里的「一层」**（一个 `TransformerBlock`），问两件事：

1. **前向时，这一层为了以后能做反向传播，额外存了多少中间结果（保存张量）？**  
   并列出其中体积最大的 5 类，各占这一层总量的百分之几。
2. **反向经过这一层时，一边释放上面那些保存张量，一边写出梯度；梯度大概占多少显存？**  
   这个数是否和「这一层参数该有多少梯度」的粗算一致？

工具上：用 Nsight Systems 录一整步训练，并在每一层外面包一层标签（NVTX），这样时间轴上能看见「现在跑到第几层」。  
层内保存张量的体积，则用 PyTorch 的 `saved_tensors_hooks` 直接数出来（讲义 §3 的同一种量法；讲义把这些保存张量也叫 residuals）。

**本问设定：** xl，batch=4，context=512，FP32，完整训练步（前向 + loss + 反向 + 优化器）。

### 2. 问题是什么？（以及我们怎么量）

需要记住的只有两点：

- **保存张量**：前向算子层 $f(x)$ 时新产生、又必须留到反向才用的中间结果（attention 分数、FFN 中间激活等）。它们不是权重，也不是「残差连接里那个被加回去的 $x$」。
- **梯度**：反向时写出的「参数该怎么改」；一层里还会有一些与激活相关的梯度。

量法分三步：

**步骤 A：看整步显存长什么样（Nsight 截图等价物）。**

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_f_nsys_cuda_mem.png" alt="memory_f_nsys_cuda_mem" width="560" />

这条曲线是 Nsight 从 CUDA 分配事件重建出来的。录制约 10.75 秒，峰值约 **67.61 GiB**。  
注意：Nsight 按「cudaMalloc 段」记账，数字会略高于前面 PyTorch 的 `memory_allocated`（约 65.5 GiB），两者不是同一把尺子，但形状一致：前向堆高、反向台阶式下降。

**步骤 B：只量「一层」为反向保存了多少。**

我们取中间一层（第 16 层）为代表。前向经过这一层时，为反向一共保存了 **1565.7 MiB**（约 1.53 GiB），共 46 个张量。  
按体积从大到小排前 5 名如下（名字是按张量形状归的类，方便理解「是哪一类东西」）：

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_f_residual_top5.png" alt="memory_f_residual_top5" width="520" />

| 排名 | 是什么 | 体积 (MiB) | 占这一层保存总量 |
|-----:|--------|----------:|-----------------:|
| 1 | FFN 中间激活（宽为 $d_{\mathrm{ff}}=10240$ 那一段） | 680.0 | 43.4% |
| 2 | Attention 的 $S\times S$ 分数 / softmax 权重 | 384.0 | 24.5% |
| 3 | 残差流 / 隐状态类激活（形状接近 $(B,S,d)$） | 360.0 | 23.0% |
| 4 | FFN 相关权重（为反传而引用的参数视图） | 100.0 | 6.4% |
| 5 | Attention 的 Q/K/V 按头拆开后的激活 | 40.0 | 2.6% |

读表时只需记住一件事：**这一层里，最大头不是「残差流那一条细细的 $(B,S,d)$」，而是 FFN 中间那段更宽的激活，以及 attention 的 $S\times S$ 矩阵。** 这和 (d)(e) 的结论是对齐的——(d) 的残差流只有 20 MiB；(e) 单份 $S\times S$ 是 128 MiB；这里一层里会存多份同类东西，所以汇总后更大。

**步骤 C：估算「一层写出的梯度」占多少显存。**

反向经过某一层时，大致同时发生两件事：

- 释放：该层前向存下的保存张量（体积记为 $R$）被丢掉；
- 分配：写出该层参数（以及一些激活侧）的梯度（体积记为 $G$）。

因此，若其它小分配大致抵消，显存的净变化满足

$$
\Delta \approx G - R \quad\Rightarrow\quad G \approx \Delta + R.
$$

我们对每一层测一次反向时的 $\Delta$，取中位数，再加上代表层的 $R$，得到

$$
G \approx 1128.7\ \text{MiB}
\quad\text{（约 1.1 GiB / 层）}.
$$

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_f_bwd_deltas.png" alt="memory_f_bwd_deltas" width="560" />

作为对照，只把「这一层参数本身」的梯度体积算一个下界（注意力 4 个 $d\times d$ 矩阵 + SwiGLU 的 3 个 $d\times d_{\mathrm{ff}}$ + 两个 LayerNorm，全是 FP32）：

$$
\bigl(4d^{2}+3d\cdot d_{\mathrm{ff}}+2d\bigr)\cdot 4/1024^{2}\approx 400\ \text{MiB}
\quad(d=2560,\ d_{\mathrm{ff}}=10240).
$$

实测 $G\approx 1129$ MiB **大于** 这个 400 MiB 下界，这是合理的：下界只数了参数梯度，实测还包含激活侧梯度、临时缓冲和分配器噪声。

### 3. 结论如何解读？

**解答 (f):**

1. **一层为反向大约要多存 1.5 GiB。**  
   第 16 层保存张量合计 **1566 MiB**。其中最大的一块是 FFN 中间激活（约 43%），其次是 attention 的 $S\times S$（约 25%），再才是残差流 / 隐状态（约 23%）。  
   所以：显存压力并不主要来自「那条细细的残差流」（概念 2，单张才 20 MiB），而来自 **$f(x)$ 内部新造、又必须留到反向的中间结果**（概念 3）——尤其是 FFN 变宽后的激活和 attention 方阵。

2. **反向时，一层写出的梯度大约 1.1 GiB。**  
   用「显存变化 = 梯度 − 释放的保存张量」反推，得 $G\approx 1129$ MiB。它不低于「只数参数梯度」的约 400 MiB 下界，方向和量级都对得上。

3. **和整网峰值怎么连起来想？**  
   前向时，32 层各自的保存张量会叠在一起，所以整网激活远大于「单层 1.5 GiB」。反向时逐层释放，显存呈台阶下降——这正是 (a) 里那条 train 曲线中间段的形状。Nsight 图只是把同一件事换了一把尺子再看一遍，并加上「按层」的标注。

产物目录：`/root/.dev/ml-sys/cs336/assignment2-systems/artifacts/memory_profiling/nsys_f/`。

## 附录：完整 8 格阶段表（FP32/BF16）

| ctx | mp | mode | peak GiB | fwd max | loss max | bwd max | opt max |
|----:|----|------|---------:|--------:|---------:|--------:|--------:|
| 128 | FP32 | forward | 18.053 | 18.053 | — | — | — |
| 128 | FP32 | train | 51.416 | 43.473 | 43.531 | 51.023 | 51.416 |
| 128 | BF16 | forward | 22.452 | 22.452 | — | — | — |
| 128 | BF16 | train | 51.406 | 47.857 | 47.910 | 51.014 | 51.406 |
| 512 | FP32 | forward | 39.750 | 39.750 | — | — | — |
| 512 | FP32 | train | 65.538 | 65.133 | 65.362 | 65.538 | 51.473 |
| 512 | BF16 | forward | 36.832 | 36.832 | — | — | — |
| 512 | BF16 | train | 62.674 | 62.217 | 62.427 | 62.674 | 51.436 |

产物路径：`/root/.dev/ml-sys/cs336/assignment2-systems/artifacts/memory_profiling/` · 报告生成时间 UTC。
