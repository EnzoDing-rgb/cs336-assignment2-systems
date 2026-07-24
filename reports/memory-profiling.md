# Memory Profiling

**设定：** `BasicsTransformerLM` **xl**；batch=4；context ∈ {128, 512}；mode ∈ {forward, train}；precision ∈ {FP32, BF16}；warmup=2；train = forward + loss + backward + `AdamW.step()`。

**关于 handout 的 128/2048（为何不用 2048）：**

讲义默认 batch $B=4$，并要求 xl 在 context $S\in\{128,2048\}$ 上做完整训练步；后续 activation checkpointing 示例也写死了形状 `[4, 2048, 2560]`，即 $B=4,\ S=2048,\ d_{\mathrm{model}}=2560$。

先解释 attention 中那块随 $S^2$ 增长的显存从何而来。在每一层中，token 首先被投影为 $Q,K,V$（每个 head 的维度为 $d_k=d_{\mathrm{model}}/H$），然后计算

$$
\mathrm{scores}=QK^{\top}/\sqrt{d_k}\in\mathbb{R}^{B\times H\times S\times S},
$$

再经 `softmax` 得到注意力权重，最后乘以 $V$。为了支持反向传播，autograd 通常需要将这份 **$S\times S$ 的 score（以及/或 softmax 后的注意力权重）逐层保存下来**。xl 的超参为 $H=32$ 头、$L=32$ 层；FP32 每元素占 4 bytes。若每层只计 **一张** score 图、且 32 层在反向传播前同时存活，则

$$
\underbrace{B}_{\text{batch}}\times\underbrace{H}_{\text{heads}}\times\underbrace{S\cdot S}_{S\times S\text{ 矩阵}}\times\underbrace{4}_{\text{bytes/elem}}\times\underbrace{L}_{\text{层数}}
=4\cdot 32\cdot 2048^{2}\cdot 4\cdot 32
=2^{36}\ \text{bytes}=64\ \text{GiB}.
$$

这里 64 GiB **并非**「整次训练只需 64 GiB」。它仅是「所有层的 attention score 缓存」这一项的下界；同一次前向中还有残差流激活 $(B,S,d)$、FFN 中间激活、往往还有第二份 $S\times S$（softmax 输出，即注意力权重）、模型参数本身（约十几 GiB），完整训练步还要再加 AdamW 的两份状态（约为参数量的两倍）。因此即便使用 80 GiB 的卡，「64 GiB 的 score 加上其它一切」也会明显超过 80 GiB。实证上，同设定 $B=4$ 在 $S=1024$ 时前向已 OOM（见 `/root/.dev/ml-sys/cs336/assignment2-systems/reports/nsys-profile.md`）。

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

**解答 (a):** 仅前向（未加载模型参数、无 Adam）时显存爬升至约 **39.750 GiB** 后进入平台期，直到释放激活。完整训练步的基线已包含模型参数和 AdamW（约 50.908 GiB），前向将激活堆叠上去，在 `forward`/`loss` 附近达到整步峰值 **65.538 GiB**；`backward` 呈台阶式下降（逐层释放为反传而暂存的中间结果），结束后驻留约 **50.985 GiB**；`optimizer` 几乎平坦（约 50.985 GiB），因为 Adam 状态早在 warmup 阶段就已分配完毕。因此：**前向=爬升/高平台，反向=台阶下降，优化器=平坦**——结合图中的阶段竖线，三段可以明确区分。

## (b) 不同 context 下的峰值显存

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_b_peaks_by_context.png" alt="peaks by context" width="560" />

| context | forward peak (GiB) | train peak (GiB) |
|--------:|-------------------:|-----------------:|
| 128 | 18.053 | 51.416 |
| 512 | 39.750 | 65.538 |

单位：`torch.cuda.max_memory_allocated` 在该次被 profile 的 step 上的全局峰值（GiB）。
训练步远大于前向的原因：完整一步除了模型参数之外，还要常驻 AdamW 状态，并在反向阶段短暂叠加上梯度；
context 变长时两者都会增长，但 attention 的 $S\times S$ 项使涨幅快于线性。

## (c) 混合精度（BF16）

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_c_bf16_vs_fp32.png" alt="BF16 vs FP32 memory" width="640" />

左图为 ctx=512 的实测峰值；右图将训练步峰值拆分为「模型参数 / Adam（始终为 FP32）」与「激活（仅 autocast 时才可能变窄）」的示意，说明即便激活理想减半，总峰值也远不会减半。

| context | mode | FP32 peak (GiB) | BF16 peak (GiB) | Δ |
|--------:|------|----------------:|----------------:|---|
| 128 | forward | 18.053 | 22.452 | +4.399 GiB (+24.4%) |
| 128 | train | 51.416 | 51.406 | -0.009 GiB (-0.0%) |
| 512 | forward | 39.750 | 36.832 | -2.917 GiB (-7.3%) |
| 512 | train | 65.538 | 62.674 | -2.864 GiB (-4.4%) |

**解答 (c):** `torch.autocast(bf16)` 只将 **部分矩阵乘的激活** 算成 BF16；**参数本身仍为 FP32**，AdamW 的一阶/二阶动量也仍为 FP32。因此峰值显存里「模型参数 + 优化器状态」这一大块几乎不动——完整训练步的基线就已约 50 GiB（见 (a)），真正可能变窄的只有激活。以 xl·ctx=512 为例：前向峰值 39.750→36.832 GiB（少约 2.9 GiB，−7.3%），训练步峰值 65.538→62.674 GiB（少约 2.9 GiB，−4.4%）；ctx=128 的训练步几乎不变（51.416→51.406 GiB），前向甚至因临时 dtype 转换略增（18.053→22.452 GiB）。换句话说，激活即便理想减半，也只能从总峰值里抠出几个 GiB，绝不可能把 65 GiB 压到 30 GiB 量级。算力侧则不同：BF16 能走 Tensor Core，端到端 step 时间往往能明显缩短（见 `/root/.dev/ml-sys/cs336/assignment2-systems/reports/mixed-precision.md`）；显存峰值这边，对本设定的收益就是「少几个 GiB」，不足以靠混精单独解决 OOM。

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

**解答 (e):** 最大的单次分配均为 **128.0 MiB**。其来源与 (d) 不同：这里不是残差流 $(B,S,d)$，而是 attention 中 $QK^{\top}$ 得到的 score（或随后的 softmax 注意力权重），形状为 $(B,H,S,S)=(4,32,512,512)$。体积为

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

- **保存张量**：前向算子层 $f(x)$ 时新产生、又必须留到反向才用的中间结果（attention 分数、FFN 中间激活等）。它们不是模型参数，也不是「残差连接里那个被加回去的 $x$」。
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
| 1 | FFN 中间激活：单张 $(B,S,d_{\mathrm{ff}})=4\cdot512\cdot10240\cdot4/1024^{2}=80$ MiB；本桶合计 $680=8.5\times80$（SwiGLU 的 $w_1x$、$w_3x$、`silu`、逐元乘积等会被多次保存） | 680.0 | 43.4% |
| 2 | Attention 的 $S\times S$：单张 $(B,H,S,S)=4\cdot32\cdot512\cdot512\cdot4/1024^{2}=128$ MiB；本桶 $384=3\times128$（典型是 score / mask 后 / softmax 注意力权重） | 384.0 | 24.5% |
| 3 | 残差流 / 隐状态：单张 $(B,S,d)=4\cdot512\cdot2560\cdot4/1024^{2}=20$ MiB；本桶 $360=18\times20$（层内多处 $(B,S,d)$ 激活被为反传保存） | 360.0 | 23.0% |
| 4 | FFN 的 $W_1$ / $W_2$ / $W_3$ 之一（常驻参数；hooks 按形状记 100 MiB，**不是新分配**，见附录 §8.0） | 100.0 | 6.4% |
| 5 | Attention 投影后的 $Q$ / $K$ / $V$（按 32 头拆开后的激活；本层保存 **2 份**，各 $(B,H,S,d_k)$、20 MiB） | 40.0 | 2.6% |

读表时只需记住一件事：**这一层里，最大头不是「残差流那一条细细的 $(B,S,d)$」，而是 FFN 中间那段更宽的激活，以及 attention 的 $S\times S$ 矩阵。** 这和 (d)(e) 的结论是对齐的——(d) 的残差流只有 20 MiB；(e) 单份 $S\times S$ 是 128 MiB；这里一层里会存多份同类东西，所以汇总后更大。

**步骤 C：$R$ 与 $G$ 到底是什么（以及为什么 $G<R$）。**

先把时间线钉死。完整训练步是：**整网前向先跑完 → 再整网反向**。对「某一层」来说：

**$R$：前向阶段留下来的。**  
前向穿过这一层时，autograd 会把一批中间结果留在显存里，专门留给以后反向用——attention 的 $S\times S$、FFN 中间激活、若干 $(B,S,d)$ 等等。这些东西在前向结束时已经占着坑，反向还没开始。我们用 `saved_tensors_hooks` 把它们的体积加总，得到

$$
R \approx 1566\ \text{MiB}.
$$

所以：$R$ **不是梯度**，也 **不是** 常驻参数本身（参数单独占约 38 GiB 地板，见 §2、附录 §8.0）；$R$ 是「为了将来能算出梯度，而在前向提前寄存的 **中间激活**（及少量对参数的引用登记）」。

**$G$：反向阶段新写出来的。**  
反向扫到这一层时，才真正去算「参数该怎么改、激活该怎么回传」。算出来的那些**梯度张量**会新占显存——主要是这一层各个参数的 `.grad`（形状通常和参数一样），再加上一些激活侧的梯度。我们把这一层反向过程中新出现的、属于「梯度」这一类的显存记为 $G$。

所以：$G$ **就是梯度占用的显存**——主要是参数的 `.grad`（与 $W_q$ 等同形的张量，见 §8.0），加上反向里短暂的激活梯度。  
**常驻参数 $W$ 不在 $G$ 里；$W.\mathrm{grad}$ 在 $G$ 里。**  
「为了算梯度而提前存下的前向中间结果」是 $R$，不是 $G$：前向存 $R$，反向用 $R$ 算出 $G$，用完释放 $R$。

用一个更日常的比喻：

- $R$＝做菜前备好的食材（占案板）；
- $G$＝炒完装盘的成品（占碗）；
- 反向这一层＝边用食材边出菜，案板上的食材撤走、碗里的菜端上来。

**为什么计数器看不到单独的 $G$，只能看 $\Delta$？**  
反向扫过一层的那一小段时间里，两件事几乎同时发生：食材撤走（$-R$），菜端上来（$+G$）。`memory_allocated` 只能告诉你案板+碗的总变化：

$$
\Delta \;=\; \text{（层后占用）}-\text{（层前占用）}
\;\approx\; (+G) + (-R) \;=\; G - R.
$$

于是 $G\approx\Delta+R$。前向单独量 $R$，反向量 $\Delta$，两者合起来才得到 $G$——这就是讲义要求的算法，不是凭空套公式。

实测各层 $\Delta\approx-437$ MiB（下图）。负号的意思很具体：**撤走的 $R$ 比端上来的 $G$ 更大**，所以反向时整卡显存在往下掉。代入：

$$
G \approx \Delta + R \approx (-437) + 1566 = 1129\ \text{MiB}.
$$

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_f_bwd_deltas.png" alt="memory_f_bwd_deltas" width="560" />

**为什么 $G$ 往往比 $R$ 小？**  
因为两者装的不是一类东西。$R$ 里尽是又大又临时的激活（单份 $S\times S$ 就 128 MiB，FFN 中间一张 80 MiB，一层里还存多份），它们只是「算梯度时要用的草稿纸」；$G$ 的主体是参数梯度，一层参数梯度的解析下界大约只有

$$
\bigl(4d^{2}+3d\cdot d_{\mathrm{ff}}+2d\bigr)\cdot 4/1024^{2}
=\bigl(4\cdot2560^{2}+3\cdot2560\cdot10240+2\cdot2560\bigr)\cdot4/1024^{2}
\approx 400\ \text{MiB}.
$$

实测 $G\approx1129$ MiB 大于这 400 MiB，是因为还含激活侧梯度和分配器临时块；但它仍然小于 $R\approx1566$ MiB——草稿纸（激活）通常比最终写下来的答案（梯度）更大。所以 $G<R$、$\Delta<0$，和「反向台阶下降」是同一件事。

### 3. 结论如何解读？

**解答 (f):**

1. **一层为反向大约要多存 1.5 GiB（这是 $R$）。**  
   第 16 层保存张量合计 **1566 MiB**。其中最大的一块是 FFN 中间激活（约 43%），其次是 attention 的 $S\times S$（约 25%），再才是残差流 / 隐状态（约 23%）。  
   所以：显存压力并不主要来自「那条细细的残差流」（概念 2，单张才 20 MiB），而来自 **$f(x)$ 内部新造、又必须留到反向的中间结果**（概念 3）——尤其是 FFN 变宽后的激活和 attention 方阵。

2. **反向时，一层新写出的梯度大约 1.1 GiB（这是 $G$）。**  
   $G$ 不是又一批「为梯度而存的中间变量」（那是 $R$）；$G$ 是反向算出来的梯度张量本身。净变化 $\Delta\approx-437$ MiB，故 $G\approx\Delta+R\approx1129$ MiB。它高于「只数参数梯度」的约 400 MiB 下界，又小于 $R$，所以反向显存会下降。

3. **和整网峰值怎么连起来想？**  
   前向时，32 层各自的 $R$ 会叠在一起，所以整网激活远大于「单层 1.5 GiB」。反向时逐层用掉并释放 $R$、写出较小的 $G$，显存呈台阶下降——这正是 (a) 里那条 train 曲线中间段的形状。

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

## 附录：符号表

| 符号 | 指什么 | 本报告里的取值 / 例子 |
|------|--------|----------------------|
| $B$ | batch size（一次喂进模型的序列条数） | $B=4$ |
| $S$ | context / 序列长度（每条有多少 token） | 主实验 $S\in\{128,512\}$；讲义还写过 $2048$ |
| $H$ | attention 头数 | xl：$H=32$ |
| $L$ | Transformer 层数（`TransformerBlock` 个数） | xl：$L=32$ |
| $d$ 或 $d_{\mathrm{model}}$ | 隐藏宽度 / 残差流最后一维 | xl：$d=2560$ |
| $d_{\mathrm{ff}}$ | FFN（SwiGLU）中间宽度 | xl：$d_{\mathrm{ff}}=10240$ |
| $d_k$ | 每个 attention head 的维度 | $d_k=d/H=2560/32=80$ |
| $Q,K,V$ | 查询 / 键 / 值投影 | 形状多为 $(B,H,S,d_k)$，如 $(4,32,512,80)$ |
| $\mathrm{scores}$ | $QK^{\top}/\sqrt{d_k}$ 得到的注意力分数（softmax **之前**） | 形状 $(B,H,S,S)$，如 $(4,32,512,512)$；单张 FP32 $=128$ MiB |
| 注意力权重 | softmax 归一化后的 $(B,H,S,S)$ 矩阵；**不是**模型参数 | 与 scores 同形状；下文凡指此项均写全「注意力权重」，不与「参数」混称 |
| 参数 | 可学习的 `nn.Parameter`（$W_q$、Embedding、SwiGLU 矩阵等） | 全文称「参数」或「模型参数」，不用简称「权重」 |
| $x$ | 残差连接里子层的输入（主干上的激活） | 形状 $(B,S,d)$；加法 $x+x_{\mathrm{attn}}$ 里是同一块显存的引用 |
| $f(x)$ | 子层变换（attention 或 FFN），不含旁路加法 | 其**内部**中间结果才是「保存张量」的主要来源 |
| $y$ | 残差连接输出 $y=x+f(x)$ | 仍是 $(B,S,d)$，写回残差流 |
| $R$ | **前向**为反传寄存的保存张量总体积（不是梯度） | 约 $1566$ MiB；含 $S\times S$、FFN 中间激活等「草稿纸」 |
| $G$ | **反向**新写出的梯度张量总体积（不是又一批保存的中间变量） | 约 $1129$ MiB；主体是参数 `.grad` + 部分激活梯度 |
| $\Delta$ | 反向扫过某一层前后显存净变化 | $\Delta\approx G-R\approx-437$ MiB；$G<R$ 故为负 |
| FP32 / BF16 | 单精度 4 bytes/元素；脑浮点 2 bytes/元素 | 模型参数与 Adam 状态在混精下仍多为 FP32 |
| MiB / GiB | $1024^{2}$ / $1024^{3}$ 字节 | 残差流单张 $20$ MiB $=20/1024$ GiB |

## 附录：理论架构与资源账本（细颗粒度）

> **读法：** 本附录建立的是 **FP32 理论账本**，公式用本报告符号（见上文符号表）。默认代入  
> $B=4,\ S=512,\ V=10000,\ d=2560,\ d_{\mathrm{ff}}=10240,\ H=32,\ L=32,\ d_k=80$，每元素 $4$ bytes。  
> **FLOPs** 对矩阵乘按常见约定计为 $2\cdot m\cdot n\cdot k$（一次乘加算两次浮点运算）。  
> **Memory** 分四类记账，不要混：  
> (1) **参数**（模型参数，常驻）；(2) **激活 / 保存张量 $R$**（前向产出、反传前常需存活）；(3) **梯度 $G$**（反向新写）；(4) **Optimizer 状态**（AdamW 的一阶/二阶矩，常驻）。  
> 实验峰值会因分配器对齐、临时缓冲、实现细节而与理论不同——**本节不把实验数硬套进公式**；文末只留一句对照。

### 1. 完整架构图（`BasicsTransformerLM` · xl）

与 `cs336_basics/model.py` 一致：pre-norm；RoPE 加在 $Q,K$ 上；FFN 为 SwiGLU（三个线性：$w_1,w_3$ 升维，$w_2$ 降维）；`lm_head` **未**与 token embedding 绑权（源码中绑权被注释掉）。

图由脚本生成（可复现）：

```bash
uv run --no-sync python scripts/plot_transformer_architecture.py
```

<img src="/root/.dev/ml-sys/cs336/assignment2-systems/reports/figures/memory_architecture_basics_transformer_lm.png" alt="BasicsTransformerLM architecture" width="640" />

**数据怎么流：** Token ID 查表变成残差流向量 → 每一层先 LN 再算 Attention，加回残差流 → 再 LN 再算 SwiGLU，再加回残差流 → 如此 $L$ 次 → 最终 LN → 线性映到词表得到 logits。  
**残差连接在图里的位置：** Attention / FFN 子层末尾的残差加法就是 $y=x+f(x)$（概念 1）；竖着贯穿各层的 $(B,S,d)$ 就是残差流（概念 2）；Attention/FFN **框内部**为反传留下的中间张量才是保存张量 $R$（概念 3）。

---

### 2. 静态资源：参数与 Optimizer（与 $S$ 无关）

参数体积 $=\#\mathrm{params}\times 4$ bytes。AdamW 通常为每个参数再存两份 FP32 状态（一阶矩 $m$、二阶矩 $v$），故 Optimizer 理论显存 $\approx 2\times$ 参数显存。参数本身在训练中也要常驻，故「参数 + Adam」常驻下界 $\approx 3\times$ 参数显存。

| 模块 | 参数形状 / 个数公式 | 代入 xl 的元素数 | 参数显存 (MiB) | AdamW 两份状态 (MiB) | 参数+Adam 常驻 (MiB) | 说明 |
|------|---------------------|-----------------:|---------------:|---------------------:|---------------------:|------|
| Token Embedding | $(V,d)=V\cdot d$ | $10000\cdot2560$ | $97.66$ | $195.31$ | $292.97$ | 查表；无矩阵乘 FLOPs |
| 每层 $q\_proj$ | $(d,d)=d^{2}$ | $2560^{2}$ | $25.00$ | $50.00$ | $75.00$ | 四套同尺寸投影之一 |
| 每层 $k\_proj$ | $d^{2}$ | $2560^{2}$ | $25.00$ | $50.00$ | $75.00$ | |
| 每层 $v\_proj$ | $d^{2}$ | $2560^{2}$ | $25.00$ | $50.00$ | $75.00$ | |
| 每层 $output\_proj$ | $d^{2}$ | $2560^{2}$ | $25.00$ | $50.00$ | $75.00$ | |
| 每层 SwiGLU $w_1$ | $(d,d_{\mathrm{ff}})$ | $2560\cdot10240$ | $100.00$ | $200.00$ | $300.00$ | 升维 |
| 每层 SwiGLU $w_3$ | $(d,d_{\mathrm{ff}})$ | $2560\cdot10240$ | $100.00$ | $200.00$ | $300.00$ | gate 支路 |
| 每层 SwiGLU $w_2$ | $(d_{\mathrm{ff}},d)$ | $10240\cdot2560$ | $100.00$ | $200.00$ | $300.00$ | 降维 |
| 每层 RMSNorm $\times2$ | $2\cdot d$ | $2\cdot2560$ | $0.020$ | $0.039$ | $0.059$ | 可学习 scale |
| **一层小计** | $4d^{2}+3d\cdot d_{\mathrm{ff}}+2d$ | $1.0486\times10^{8}$ | **400.02** | **800.04** | **1200.06** | |
| **$L=32$ 层** | $\times L$ | | **12800.6** | **25601.2** | **38401.8** | |
| `ln_final` | $d$ | $2560$ | $0.010$ | $0.020$ | $0.029$ | |
| LM Head | $(d,V)=d\cdot V$ | $2560\cdot10000$ | $97.66$ | $195.31$ | $292.97$ | 与 Embedding **未绑权**，故另计一份 |
| **全模型合计** | | | **$\approx12996$ MiB $\approx12.69$ GiB** | **$\approx25.38$ GiB** | **$\approx38.07$ GiB** | 理论常驻下界（尚未含任何激活/$R$/$G$） |

**为什么这样算：** Embedding / LM Head 是 $V\times d$ 的大表；Attention 四个 $d\times d$；SwiGLU 三个 $d\times d_{\mathrm{ff}}$（宽四倍，$d_{\mathrm{ff}}=4d$），所以 **FFN 参数量大于 Attention 参数量**——这也解释了为何单层保存张量里 FFN 相关块往往更大。  
RoPE 的 $\cos/\sin$ 缓存是 buffer，一般不当作 Adam 参数，此处不计入上表。

---

### 3. 单层 · 前向：算子 × 计算量 × 激活显存 × 理论 $R$ 贡献

约定（本节一律用 $B{=}4,\ S{=}512,\ d{=}2560,\ d_{\mathrm{ff}}{=}10240,\ H{=}32,\ d_k{=}80$，FP32 $=4$ bytes/元素）：

**显存怎么算：** $\text{MiB}=\dfrac{\#\mathrm{elements}\times 4}{1024^{2}}$。

**五种常见张量（20 / 25 / 80 / 100 / 128 不是一回事）：**

| 是什么 | 典型形状 | 公式 | 代入 xl | 精确值 (MiB) |
|--------|----------|------|---------|-------------:|
| 残差流形激活（也含按头拆分后的 $Q,K,V$） | $(B,S,d)$ 或 $(B,H,S,d_k)$ | $\dfrac{B\!\cdot\! S\!\cdot\! d\!\cdot\! 4}{1024^{2}}$（因 $H\!\cdot\! d_k=d$，按头拆分后与 $(B,S,d)$ **同字节**） | $\dfrac{4\cdot512\cdot2560\cdot4}{1024^{2}}=\dfrac{20{,}971{,}520}{1{,}048{,}576}$ | **20.0** |
| $d\!\times\! d$ 的参数矩阵（或它的 `.grad`） | $(d,d)$，如 $W_q,W_o$ | $\dfrac{d^{2}\!\cdot\! 4}{1024^{2}}$ | $\dfrac{2560^{2}\!\cdot\! 4}{1024^{2}}=\dfrac{6{,}553{,}600\cdot4}{1{,}048{,}576}$ | **25.0** |
| FFN 中间宽激活 | $(B,S,d_{\mathrm{ff}})$ | $\dfrac{B\!\cdot\! S\!\cdot\! d_{\mathrm{ff}}\!\cdot\! 4}{1024^{2}}$ | $\dfrac{4\cdot512\cdot10240\cdot4}{1024^{2}}$ | **80.0** |
| $d\!\times\! d_{\mathrm{ff}}$ 的参数矩阵（或它的 `.grad`） | $(d,d_{\mathrm{ff}})$，如 $W_1$ | $\dfrac{d\!\cdot\! d_{\mathrm{ff}}\!\cdot\! 4}{1024^{2}}$ | $\dfrac{2560\cdot10240\cdot4}{1024^{2}}$ | **100.0** |
| $S\!\times\! S$ 的注意力矩阵（scores 或注意力权重） | $(B,H,S,S)$ | $\dfrac{B\!\cdot\! H\!\cdot\! S^{2}\!\cdot\! 4}{1024^{2}}$ | $\dfrac{4\cdot32\cdot512^{2}\cdot4}{1024^{2}}$ | **128.0** |

**20 与 25 的本质区别：** 残差流形激活带 batch、序列维（$B\!\cdot\! S=2048$ 个 token 位置），每位置 $d$ 个数 → $5{,}242{,}880$ 元素；而 $d\!\times\! d$ 参数矩阵是 **单个** 矩阵 → $6{,}553{,}600$ 元素。后者多 $25\%$，所以是 **25.0 MiB**，不是 20 MiB。

- **本算子新增显存**＝**仅此一步**向 GPU **新申请**的 buffer。若输出只是 view / 引用 / 原地写回 → **0**。
- **理论 $R$ 增量**＝此算子让 autograd **额外**保存、且**前面行尚未计入**的中间结果。
- 残差加法几乎不增参数；理想实现可原地写回残差流（概念 1），但 $f(x)$ **内部**各 matmul / $S\times S$ 仍是新分配。

| # | 算子 | 计算（前向在干什么） | 形状（主输出） | 理论 FLOPs（代入数） | 本算子新增显存 (MiB)：公式 → 值 | 理论 $R$ 增量 (MiB) | 备注 |
|--:|------|----------------------|----------------|---------------------:|--------------------------------:|--------------------:|------|
| F1 | `ln1` RMSNorm | 按最后一维：$x/\mathrm{rms}(x)\cdot\gamma$ | $(B,S,d)$ | $\sim O(BSd)$ | **+20.0**（$BSD\!\cdot\!4/1024^{2}$） | **+20.0**（存 $x$，同形） | 每层两次 |
| F2 | `q_proj` | $Y=XW_q$；输出 `view` 成 $(B,H,S,d_k)$ | $(B,H,S,d_k)$ | $2.684\times10^{10}$ | **+20.0**（$H d_k=d$，与 $(B,S,d)$ 同字节） | **+20.0**（存 $X$） | 三投影同构 |
| F3 | `k_proj` | 同 F2，$W_k$ | $(B,H,S,d_k)$ | 同 F2 | **+20.0** | **+20.0** | |
| F4 | `v_proj` | 同 F2，$W_v$ | $(B,H,S,d_k)$ | 同 F2 | **+20.0** | **+20.0** | |
| F5 | RoPE | 对 $Q,K$ 旋转 | 各 $(B,H,S,d_k)$ | $\sim O(BHSd_k)$ | **+40.0**（$Q,K$ 各一份；原地 **0.0**） | **+20.0**–**+40.0** | |
| F6 | $QK^{\top}/\sqrt{d_k}$ | 每 head：$QK^{\top}$ | $(B,H,S,S)$ | $5.369\times10^{9}$ | **+128.0**（$BHS^{2}\!\cdot\!4/1024^{2}$） | **+128.0** | 随 $S^{2}$ 涨 |
| F7 | causal mask | 原地写回 scores | $(B,H,S,S)$ | $O(BHS^{2})$ | **0.0** | **0.0** | 逻辑仍 $128$ MiB |
| F8 | softmax | 原地写回 | $(B,H,S,S)$ | $\sim O(BHS^{2})$ | **0.0** | **0.0** 或 **+128.0** | 注意力权重 |
| F9 | 注意力权重 $\times V$ | $\mathrm{Attn}\,V$ | $(B,H,S,d_k)$ | $5.369\times10^{9}$ | **+20.0** | **0.0** | |
| F10a | merge（拼头） | $(B,H,S,d_k)\!\to\!(B,S,d)$ | $(B,S,d)$ | $0$ | **0.0**（`view`） | **0.0** | 见 **F10 详解** |
| F10b | `output_proj` | $Y=XW_o$ | $(B,S,d)$ | $2.684\times10^{10}$ | **+20.0** | **+20.0**（Linear 输入） | $W_o$ 是 $25.0$ MiB 参数，常驻 |
| F11 | 残差 ADD1 | $y=x+x_{\mathrm{attn}}$ | $(B,S,d)$ | $O(BSd)$ | **0.0** | **0.0** | 默认实现可能 **+20.0** |
| F12 | `ln2` | 同 F1 | $(B,S,d)$ | $\sim O(BSd)$ | **+20.0** | **+20.0** | |
| F13 | SwiGLU $w_1$ | $XW_1$ | $(B,S,d_{\mathrm{ff}})$ | $1.074\times10^{11}$ | **+80.0** | **+80.0** | |
| F14 | SwiGLU $w_3$ | $XW_3$ | $(B,S,d_{\mathrm{ff}})$ | 同 F13 | **+80.0** | **+80.0** | |
| F15 | `silu` | $u\cdot\sigma(u)$ | $(B,S,d_{\mathrm{ff}})$ | $\sim O(BSd_{\mathrm{ff}})$ | **+80.0** 或 **0.0** | **+80.0** | |
| F16 | 逐元乘 | $\mathrm{silu}(w_1)\odot w_3$ | $(B,S,d_{\mathrm{ff}})$ | $\sim O(BSd_{\mathrm{ff}})$ | **+80.0** 或 **0.0** | **+80.0**–**+160.0** | |
| F17 | SwiGLU $w_2$ | $ZW_2$ | $(B,S,d)$ | $1.074\times10^{11}$ | **+20.0** | **+80.0**（存 $Z$，宽激活） | |
| F18 | 残差 ADD2 | $x'+x_{\mathrm{ffn}}$ | $(B,S,d)$ | $O(BSd)$ | **0.0** | **0.0** | 层输出 |

**F10 详解（拼头 + `output_proj`）：**

1. **F9 输出** $(B,H,S,d_k)$：元素数 $B\!\cdot\! H\!\cdot\! S\!\cdot\! d_k=B\!\cdot\! S\!\cdot\! d$，即 **20.0 MiB**。
2. **拼头（F10a）**：$(B,H,S,d_k)\to(B,S,d)$，纯 `view` 时新增 **0.0 MiB**。
3. **`output_proj`（F10b）**：Linear 必开新输出，新增 **+20.0 MiB**。参数 $W_o$ 是 $d\!\times\! d$ 矩阵（**25.0 MiB**）常驻，不计入本步激活增量。
4. **为何 multi-head 要这样设计**：先把 $d$ 维拆成 $H$ 个子空间并行做注意力（不同 head 学不同模式），拼头后用 $W_o$ 把各 head 混回统一的残差流表示。
5. **`cs336_basics` 实现细节**：`model.py` 在拼头后调用了 `.contiguous()`，会**强制拷贝**一份 $(B,S,d)$，实现上 F10a **额外 +20 MiB**（不在上表「理想 view」账里，但跑本仓库时要记住）。

**单层前向 matmul 主导 FLOPs（把 F2–F4、F6、F9、F10b、F13、F14、F17 的代入式相加）：**

$$
\begin{aligned}
&\underbrace{3\cdot 2(BS)d^{2}}_{\text{QKV}=3\times2.684\times10^{10}}
+\underbrace{2BHS^{2}d_k}_{QK^{\top}=5.369\times10^{9}}
+\underbrace{2BHSd_kS}_{\mathrm{Attn}V=5.369\times10^{9}}
+\underbrace{2(BS)d^{2}}_{W_o\ \text{(F10b)}=2.684\times10^{10}}\\
&+\underbrace{2\cdot 2(BS)d\,d_{\mathrm{ff}}}_{w_1,w_3=2\times1.074\times10^{11}}
+\underbrace{2(BS)d_{\mathrm{ff}}d}_{w_2=1.074\times10^{11}}
\approx 4.40\times 10^{11}\ \text{FLOPs/层}
\approx 0.44\ \text{TFLOP/层}.
\end{aligned}
$$

**为什么 FFN 算力压过 Attention：** $w_1,w_3,w_2$ 三项合计 $6(BS)d\,d_{\mathrm{ff}}$。代入 $d_{\mathrm{ff}}=4d$：$6(BS)d(4d)=24(BS)d^{2}=24\cdot2048\cdot2560^{2}=4.027\times10^{11}$。而 QKV+Out 只是 $8(BS)d^{2}=8\cdot2048\cdot2560^{2}=1.342\times10^{11}$。$QK^{\top}$ 与 $\mathrm{Attn}V$ 合计 $1.074\times10^{10}$，在 $S=512$ 时尚小于 FFN；但它们 $\propto S^{2}$，长 context 时会追上甚至反超。

**理论 $R$ 数量级（单层，结构账）：**  
至少 **1** 份 $S\times S$（F6 scores；F7/F8 若原地则与之间共享，不另算第二份）$=128$ MiB；FFN 中间若干份 $(B,S,d_{\mathrm{ff}})$（F13–F17）$\sim4$–$8\times80=320$–$640$ MiB；再加若干 $(B,S,d)$ / QKV $\sim$ 数百 MiB。总和 **$\sim1$–$2$ GiB/层** 合理——与实验 $R\approx1.53$ GiB 同量级，但不是把 1566 MiB 拆回填表。

---

### 4. 单层 · 反向传播：需要什么、算什么、占多少显存

这一节把 **§3 每一行前向算子** 在反向时逐一对应起来。读表前，先把三件事说清楚。

#### 4.1 反向传播在干什么？

训练时 loss 是一个标量。反向传播的目的：算出 **每个参数** 该往哪个方向改（写入 `参数名.grad`），并把 **「loss 对每个中间激活的敏感度」** 一层层传回更靠近输入的地方，以便继续算更前面参数的梯度。

可以把它想成 **倒着走一遍计算图**，但每一步做的不是「再算一遍前向」，而是 **用前向时存下的中间结果（$R$）**，结合 **从 loss 一侧传回来的梯度**，按链式法则算出本步需要的梯度。

**从 loss 一侧传回来的梯度**（下文简称 **传出梯度**）：更靠近 loss 的算子已经算好的、对本算子 **输出张量** 的梯度。例如 `output_proj` 的输出是 $(B,S,d)$，反向时我们手里先有的是「loss 对这个 $(B,S,d)$ 输出的梯度」——它从 FFN 后面的层、乃至 loss 一路传回来，并不是凭空出现的。

**本步新写出的梯度**分两类：

1. **参数梯度**：形状与参数相同，累写进 `.grad`（例如 $\partial W_o$：$d^{2}\!\cdot\!4/1024^{2}=25.0$ MiB）。
2. **激活梯度**：对中间激活的梯度（例如 $\partial L/\partial X$：$B\!\cdot\! S\!\cdot\! d\!\cdot\!4/1024^{2}=20.0$ MiB），要继续往更前面传。

#### 4.2 执行顺序：与前向相反

一层 `TransformerBlock` 的前向顺序是 F1→…→F18。反向 **从 F18 开始，倒着走到 F1**（下表 B18→B1 与此对应）。  
直觉：loss 的误差先传到本层 **最后** 一个算子（第二个残差加法的输出），再一路传回 attention 子层。

#### 4.3 几类算子的反传规则（看懂表用）

**（1）矩阵乘 $Y = XW$**（如 `q_proj`、`output_proj`、SwiGLU）

前向：输入激活 $X$，参数 $W$，输出 $Y$。  
反向时已知：传出梯度 $\partial L/\partial Y$（与 $Y$ 同形），以及前向保存的 $X$（在 $R$ 里）。

要算：

- $\partial L/\partial W = X^{\top}(\partial L/\partial Y)$ → 写入 $W$ 的 `.grad`（参数梯度）；
- $\partial L/\partial X = (\partial L/\partial Y)\,W^{\top}$ → **激活梯度**，继续往前传。

所以一次 matmul 反传的 FLOPs 粗算是前向的 **约 $2$ 倍**（既要算参数梯度，又要算激活梯度）。

**（2）残差加法 $y = a + b$**（F11、F18；表中标 **ADD1**、**ADD2**）

「ADD」不是缩写黑话，就是 **加法**。ADD1 = attention 子层后的残差加（$x + x_{\mathrm{attn}}$）；ADD2 = FFN 子层后的残差加（$x' + x_{\mathrm{ffn}}$）。

前向：两路同形 $(B,S,d)$ 相加得 $y$。  
反向：已知 $\partial L/\partial y$。因为 $y=a+b$，对 $a$ 和 $b$ 求偏导都是 $1$，所以

$$
\frac{\partial L}{\partial a} = \frac{\partial L}{\partial y}, \qquad
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y}.
$$

也就是说：**把同一份传出梯度分别交给加法的两个输入**，继续往前传。加法 **没有参数**，不产生 `.grad`；也 **不必** 为这一步单独再申请一块 $(B,S,d)$ 的新显存（实现上常是传递引用）。  
前向若把 $a,b$ 存进了 $R$，反向会 **读取** 它们，但那是前向已占的坑，不是本步新增。

**（3）LayerNorm / RMSNorm**

需要前向的输入 $x$（及实现相关的统计量）。反传写出 $\partial L/\partial x$（继续往前传）和 $\partial L/\partial\gamma$（可学习 scale 的参数梯度，很小）。

**（4）reshape / 拼头（merge）**

前向只是改张量视图，没有新矩阵乘。反向把传出梯度的形状 **改回** 前向输入时的形状即可（例如 $(B,S,d)\to(B,H,S,d_k)$），**不单独占大块新显存**。

**（5）softmax、$QK^\top$、注意力权重 $\times V$**

需要前向留下的 $S\times S$ 块（scores 或注意力权重）以及 $Q,K,V$。反传会在 $(B,H,S,S)$ 上 **短暂** 出现梯度缓冲（约 $128$ MiB），算完传给更前面的投影层。

#### 4.4 表格怎么读（记账约定）

与 §3 一致；**新增显存**列一律用 §3 五种张量的公式代入，写出 **至少一位小数** 的精确值。

**反向时两类梯度各占多少显存：**

| 写出什么 | 形状 | 公式 → 代入 → 结果 |
|----------|------|-------------------|
| 参数 `.grad`（$d\!\times\! d$，如 $\partial W_q,\partial W_o$） | $(d,d)$ | $d^{2}\!\cdot\!4/1024^{2}=2560^{2}\!\cdot\!4/1024^{2}$ → **25.0 MiB** |
| 参数 `.grad`（$d\!\times\! d_{\mathrm{ff}}$，如 $\partial W_1$） | $(d,d_{\mathrm{ff}})$ | $d\!\cdot\! d_{\mathrm{ff}}\!\cdot\!4/1024^{2}=2560\!\cdot\!10240\!\cdot\!4/1024^{2}$ → **100.0 MiB** |
| 激活梯度（残差流形 $(B,S,d)$） | $(B,S,d)$ | $B\!\cdot\! S\!\cdot\! d\!\cdot\!4/1024^{2}=4\!\cdot\!512\!\cdot\!2560\!\cdot\!4/1024^{2}$ → **20.0 MiB** |
| 激活梯度（FFN 宽形） | $(B,S,d_{\mathrm{ff}})$ | $B\!\cdot\! S\!\cdot\! d_{\mathrm{ff}}\!\cdot\!4/1024^{2}$ → **80.0 MiB** |
| 激活梯度（$S\!\times\! S$） | $(B,H,S,S)$ | $B\!\cdot\! H\!\cdot\! S^{2}\!\cdot\!4/1024^{2}$ → **128.0 MiB** |
| RMSNorm 的 $\partial\gamma$ | $(d,)$ | $d\!\cdot\!4/1024^{2}=2560\!\cdot\!4/1024^{2}$ → **0.0 MiB**（四舍五入到 0.1；精确 $0.0098$） |

- **本步新增显存**：仅本反向算子 **新向 GPU 申请** 的梯度 buffer；改形状传回（merge）或加法两路传同一引用 → **0.0**。
- **需要的前向保存物**：前向记入 $R$、反向时要读出的张量；前面行已计过的不重复加总。

---

#### 4.5 逐算子反向表（B18 → B1）

| # | 对应前向 | 反向在算什么（完整逻辑） | 需要的前向保存物 | 本步新写出 | 本步新增显存 (MiB)：公式 → 代入 → 值 | 粗算 FLOPs |
|--:|----------|--------------------------|------------------|------------|--------------------------------------:|-----------:|
| B18 | F18 残差 ADD2：$y=x'+x_{\mathrm{ffn}}$ | 已知 $\partial L/\partial y$（从下一层传回）。令 $\partial L/\partial x'=\partial L/\partial y$，$\partial L/\partial x_{\mathrm{ffn}}=\partial L/\partial y$。无参数 | $x'$、$x_{\mathrm{ffn}}$ 各 $(B,S,d)$，各 20.0 MiB（常已在 $R$） | 无 `.grad`；两路激活梯度 | **0.0**（加法不传新块；梯度传引用） | $O(BSd)$ |
| B17 | F17 $Y=ZW_2$ | $\partial W_2=Z^{\top}(\partial L/\partial Y)$；$\partial L/\partial Z=(\partial L/\partial Y)W_2^{\top}$ | $Z$：$(B,S,d_{\mathrm{ff}})$，80.0 MiB | $\partial W_2$；$\partial L/\partial Z$ | **+100.0**（$dd_{\mathrm{ff}}\!\cdot\!4/1024^{2}$）$+$ **+80.0**（$BSd_{\mathrm{ff}}\!\cdot\!4/1024^{2}$）$=$ **+180.0** | $\sim2\times1.074\times10^{11}$ |
| B16 | F16 逐元乘 | 梯度按乘积法则分回两输入 | 两路 $(B,S,d_{\mathrm{ff}})$，各 80.0 MiB | 两路输入激活梯度 | **0.0**–**+80.0**（可原地覆盖一路） | $O(BSd_{\mathrm{ff}})$ |
| B15 | F15 `silu` | 由 silu 输出梯度算 $\partial L/\partial u$ | $u$：$(B,S,d_{\mathrm{ff}})$，80.0 MiB | $\partial L/\partial u$ | **+80.0**（$BSd_{\mathrm{ff}}\!\cdot\!4/1024^{2}=4\!\cdot\!512\!\cdot\!10240\!\cdot\!4/1024^{2}$） | $O(BSd_{\mathrm{ff}})$ |
| B14 | F14 $Y=XW_3$ | $\partial W_3$、$\partial L/\partial X$ | 输入 $X$：$(B,S,d)$，20.0 MiB | $\partial W_3$ | **+100.0**（$dd_{\mathrm{ff}}\!\cdot\!4/1024^{2}=2560\!\cdot\!10240\!\cdot\!4/1024^{2}$） | $\sim2\times1.074\times10^{11}$ |
| B13 | F13 $Y=XW_1$ | 同 B14 结构 | 输入 $X$：$(B,S,d)$，20.0 MiB | $\partial W_1$ | **+100.0**（同上） | $\sim2\times1.074\times10^{11}$ |
| B12 | F12 `ln2` | $\partial L/\partial x$、$\partial\gamma$ | 输入 $x$：$(B,S,d)$，20.0 MiB | $\partial\gamma$；$\partial L/\partial x$ | **+20.0**（$BSD\!\cdot\!4/1024^{2}$）$+$ **0.0**（$\partial\gamma$：$d\!\cdot\!4/1024^{2}=0.0098$）$=$ **+20.0** | $O(BSd)$ |
| B11 | F11 残差 ADD1：$y=x+x_{\mathrm{attn}}$ | 同 B18：$\partial L/\partial x=\partial L/\partial x_{\mathrm{attn}}=\partial L/\partial y$ | $x$、$x_{\mathrm{attn}}$ 各 $(B,S,d)$ | 无 `.grad`；两路激活梯度 | **0.0** | $O(BSd)$ |
| B10b | F10b $Y=XW_o$ | $\partial W_o$、$\partial L/\partial X$ | Linear 输入：$(B,S,d)$，20.0 MiB | $\partial W_o$；$\partial L/\partial X$ | **+25.0**（$d^{2}\!\cdot\!4/1024^{2}=2560^{2}\!\cdot\!4/1024^{2}$）$+$ **+20.0**（$BSD\!\cdot\!4/1024^{2}$）$=$ **+45.0** | $\sim2\times2.684\times10^{10}$ |
| B10a | F10a merge 拼头 | 梯度 reshape $(B,S,d)\!\to\!(B,H,S,d_k)$ | F9 输出：$(B,H,S,d_k)$，20.0 MiB | 仅改形状 | **0.0** | $0$ |
| B9 | F9 注意力权重 $\times V$ | $\partial V$、$\partial$(注意力权重) | 注意力权重 $(B,H,S,S)$ 128.0 MiB；$V$ $(B,H,S,d_k)$ 20.0 MiB | $\partial V$；$\partial$(注意力权重) | **+20.0**（$BSD\!\cdot\!4/1024^{2}$）$+$ **+128.0**（$BHS^{2}\!\cdot\!4/1024^{2}$）$=$ **+148.0** | $\sim2\times5.369\times10^{9}$ |
| B8 | F8 softmax | scores 侧梯度 | 注意力权重：$(B,H,S,S)$，128.0 MiB | $\partial$(scores) | **0.0**–**+128.0**（原地则 0；新物化则 $BHS^{2}\!\cdot\!4/1024^{2}$） | $\sim O(BHS^{2})$ |
| B7 | F7 causal mask | 非法位置梯度置 0 | masked scores：$(B,H,S,S)$ | $\partial$(scores) | **0.0**（原地） | $O(BHS^{2})$ |
| B6 | F6 $QK^{\top}$ | $\partial Q$、$\partial K$ | $Q,K$ 各 $(B,H,S,d_k)$；scores $(B,H,S,S)$ | $\partial Q$；$\partial K$ | **+20.0**（$BSD\!\cdot\!4/1024^{2}$，$\partial Q$）$+$ **+20.0**（$\partial K$）$=$ **+40.0** | $\sim2\times5.369\times10^{9}$ |
| B5 | F5 RoPE | 逆旋转传回 $\partial Q,\partial K$ | 位置编码 | $\partial Q$；$\partial K$ | **0.0**–**+40.0**（两路各 $(B,H,S,d_k)$；原地则 0） | $O(BHSd_k)$ |
| B4 | F4 $Y=XW_v$ | $\partial W_v$、$\partial L/\partial X$ | 输入 $(B,S,d)$，20.0 MiB | $\partial W_v$ | **+25.0**（$d^{2}\!\cdot\!4/1024^{2}$） | $\sim2\times2.684\times10^{10}$ |
| B3 | F3 $Y=XW_k$ | 同 B4 | 输入 $(B,S,d)$，20.0 MiB | $\partial W_k$ | **+25.0**（$d^{2}\!\cdot\!4/1024^{2}$） | $\sim2\times2.684\times10^{10}$ |
| B2 | F2 $Y=XW_q$ | 同 B4 | 输入 $(B,S,d)$，20.0 MiB | $\partial W_q$ | **+25.0**（$d^{2}\!\cdot\!4/1024^{2}$） | $\sim2\times2.684\times10^{10}$ |
| B1 | F1 `ln1` | $\partial L/\partial x$、$\partial\gamma$ | 输入 $(B,S,d)$，20.0 MiB | $\partial\gamma$；$\partial L/\partial x$ | **+20.0**（$BSD\!\cdot\!4/1024^{2}$）$+$ **0.0**（$\partial\gamma$）$=$ **+20.0** | $O(BSd)$ |

**读这张表的顺序建议：** 从 B18 往下读到 B1，就是误差从「本层输出」一路回到「本层输入」的路径。每读一行，问自己三件事：（1）手里已有的传出梯度是什么形状？（2）前向留下了什么 saved tensor 要翻出来？（3）本步要写哪些 `.grad`、要把什么形状的激活梯度交给上一行？

---

#### 4.6 一层参数梯度合计（$G$ 的「地板」）

把上表所有 **写入 `.grad`** 的参数梯度加起来（一层内）：

$$
\begin{aligned}
&4\times\underbrace{25.0}_{d^{2}\cdot4/1024^{2}\ (d{\times}d\ \text{参数})}
\;+\;3\times\underbrace{100.0}_{dd_{\mathrm{ff}}\cdot4/1024^{2}\ (d{\times}d_{\mathrm{ff}}\ \text{参数})}
\;+\;2\times\underbrace{0.0098}_{d\cdot4/1024^{2}\ (\gamma)}\\
&= 100.0 + 300.0 + 0.02 = 400.02\ \text{MiB}.
\end{aligned}
$$

即 $(4d^{2}+3d\cdot d_{\mathrm{ff}}+2d)\cdot 4/1024^{2}=400.02$ MiB。这是 **仅参数 `.grad`** 的结构下界。

完整一层反向的 $G$ 还会更大：例如 B10b 一步就可能 **+45.0**（25.0 参数 $+$ 20.0 激活），B9 可达 **+148.0**（20.0 $+$ 128.0），B17 **+180.0**——这些是 **激活侧梯度** 与参数 `.grad` 叠在一起。正文用 $\Delta+R$ 估得 **$G\approx 1129$ MiB**，大于 400.02 MiB 是正常现象。

#### 4.7 与 Optimizer 的关系

AdamW 的一阶矩 $m$、二阶矩 $v$ 在第一次 `optimizer.step()` 之前就已按参数表分配好（§2），**不在反向过程中新开**。反向只负责把各参数的 `.grad` 填好；`optimizer.step()` 读 `.grad` 去更新参数时，显存曲线通常几乎平坦（正文 (a) 已示）。

---

### 5. 嵌入、最终 LN、LM Head、Loss（层外）

记账同 §3：**本算子新增显存**只计新 buffer；Embedding 查表输出是新张量，不是 view。

| 阶段 | 算子 | 前向计算 / FLOPs（代入） | 本算子新增显存 (MiB) | 理论 $R$ 增量 / 反传 | 参数梯度增量 (MiB) | Optimizer |
|------|------|--------------------------|---------------------:|----------------------|-------------------:|-----------|
| 输入 | Token IDs | 整数，无 FLOPs | **0**（不计入浮点激活；int64 索引 $\sim0.016$ 另计） | 不进浮点 $R$ | — | — |
| Emb | Embedding 查表 | 无 matmul；**输出** $(B,S,d)$ | **+20** | 反传按 token 累加进 embedding 行 | **+97.66**（整表 $\partial W_{\mathrm{emb}}$） | 已在 §2 |
| 最终 | `ln_final` | $\sim O(BSd)$ | **+20** | 同单层 LN：**+20** | **+0.01** | 已在 §2 |
| 输出 | LM Head | $2(BS)dV$；**输入** $(B,S,d)$，**输出** logits $(B,S,V)$ | **+78.125**（logits 新物化） | 存残差流输入 **+20** | **+97.66** | 已在 §2 |
| Loss | `cross_entropy` | $\sim O(BSV)$ | **0**（标量 loss）；内部 log-softmax 临时 **+78** 可随即释放 | logits 梯度瞬时 **+78.125** | — | 不经 Adam |

---

### 6. 全网理论汇总（把层叠起来时怎么想）

| 账本条目 | 理论公式 / 代入 | 量级 | 心智要点 |
|----------|-----------------|------|----------|
| 参数常驻 | Emb $97.66$ + $L\times400.02$ + ln_final $0.01$ + LMHead $97.66$ | $97.66+12800.6+0.01+97.66\approx12996$ MiB $\approx12.69$ GiB | 与 $S$ 无关 |
| AdamW 常驻 | $\approx2\times$ 参数 | $\approx25.38$ GiB | 与 $S$ 无关；首次 step 前分配 |
| 参数+Adam 常驻 | $\approx3\times$ 参数 | $\approx38.07$ GiB | 训练步「地板」 |
| 单层前向 matmul | §3 求和 | $\approx4.40\times10^{11}$ FLOPs $\approx0.44$ TFLOP | FFN 三项主导 |
| 全网前向 matmul（估） | $L\times0.44+$ LMHead $0.105$ | $\approx14.2$ TFLOP | 未计 softmax/LN |
| 单层理论 $R$（量级） | $1\times128$（$S\times S$）+ $n\times80$ + $m\times20$（$n\sim4$–$8$） | **$\sim1$–$2$ GiB** | 随 $S^{2}$ 与 $d_{\mathrm{ff}}$ 涨；原地 mask/softmax 不重复计第二份 $S\times S$ |
| 全网 $R$ 峰值（无 checkpoint） | 前向结束 $\sim L$ 层保存叠加 | **$\sim L\times(1\text{–}2)$ GiB** | 长 context OOM 主因之一 |
| 单层参数梯度下界 | $4\times25+3\times100+2\times0.01$ | $\approx400$ MiB | $G$ 的地板 |
| 反向时单层净效应 | 释放 $R$、写入 $G$ | 通常 $G<R$ ⇒ 显存下降 | 解释 (a) 台阶 |

**$S$ 怎么进账（代入看比例）：**

- 残差流 / FFN 中间 $\propto BSd$ 或 $BSd_{\mathrm{ff}}$：**对 $S$ 线性**。$S:512\to2048$（$\times4$）时，一张残差流 $20\to80$ MiB，一张 FFN 中间 $80\to320$ MiB。
- Attention $S\times S$：$\dfrac{B\cdot H\cdot S\cdot S\cdot4}{1024^{2}}$。$S=512$ 时 $128$ MiB；$S=2048$ 时 $4\cdot32\cdot2048\cdot2048\cdot4/1024^{2}=2048$ MiB $=2$ GiB（**单层单份**）。再 $\times L$ 与其它项，开篇不用 $S=2048$ 是结构必然，不是实验碰巧。

---

### 7. 与正文实验的一句话边界

正文 (a)–(f) 的 GiB / Top-5 / $R\approx1566$ MiB / $G\approx1129$ MiB 是 **测量值**（含分配器与实现细节）。本附录是 **结构理论账本**：§3–§5 各表区分「逻辑形状」「本算子新增显存」「$R$ 增量」，原地/共享处标 **0**，避免把同一块 buffer 重复加总。二者同量级即互相印证；逐字节对齐既不是目标，也不应当用实验数反填理论格。

---

### 8. 实例走查：第 16 层（$R\approx1.5$ GiB → $G\approx1.1$ GiB）

本节把正文 (f) 的实测数字 **按时间顺序** 串一遍。设定与实验相同：xl，$B=4$，$S=512$，FP32，完整训练步；取 **第 16 层**（32 层正中间，`saved_tensors_hooks` 与 Nsight 都量过这一层）。

#### 8.0 参数矩阵也是张量：它算不算在 $R$ / $G$ 里？

$W_q$、$W_o$、$W_1$ 等在 PyTorch 里都是 **张量**（`nn.Parameter`）。但谈 $R$ / $G$ 时，不能把「所有张量」混成一锅——要按 **它在训练里扮演什么角色** 分开记账。整步显存大致分 **四本账**（附录 §2 也有）：

| 账本 | 是什么 | 第 16 层举例 | 典型体积（本层） | 算不算进 $R$？ | 算不算进 $G$？ |
|------|--------|--------------|------------------:|:--------------:|:--------------:|
| **① 常驻参数** | 可学习的权重矩阵，训练全程在 GPU 上 | $W_q,W_k,W_v,W_o$（各 $d\times d$）；$W_1,W_2,W_3$（$d\times d_{\mathrm{ff}}$）；RMSNorm 的 $\gamma$ | 一层约 **400 MiB** 参数（32 层合计约 12.8 GiB，再加 Embedding / LM Head） | **否** | **否** |
| **② 常驻 Optimizer** | AdamW 的 $m,v$，与参数一一对应 | 同上每个参数两份 FP32 状态 | 一层约 **800 MiB** 状态（全网约 25 GiB） | **否** | **否** |
| **③ 保存张量 $R$** | 前向为反传 **额外留下** 的中间结果（主要是 **激活**） | FFN 中间 $w_1(x)$、$S\times S$ scores、多处 $(B,S,d)$ | 本层实测 **1566 MiB** | **是**（就是 $R$ 本身） | **否** |
| **④ 梯度 $G$** | 反向 **新写出来** 的梯度 | 参数的 `.grad`；激活侧 $\partial L/\partial X$ 等 | 本层估 **1129 MiB** | **否** | **是**（就是 $G$ 本身） |

**结论先说清楚：**

- **参数矩阵本身（$W_q$、$W_o$……）既不在 $R$ 里，也不在 $G$ 里。** 它们属于账本 ①，从训练开始就一直占着（加上账本 ② 的 Adam，全文约 **38 GiB 地板**）。$R\approx1.5$ GiB、$G\approx1.1$ GiB 都是 **在这一层地板之上** 额外讨论的量，**没有把 38 GiB 算进去**。
- **参数矩阵的梯度（`W_q.grad` 等）在 $G$ 里。** 形状与参数相同：$d\times d$ 的是 25.0 MiB，$d\times d_{\mathrm{ff}}$ 的是 100.0 MiB；一层合计 **400.02 MiB**（§8.4）。
- **前向中间激活（FFN 的 80 MiB 宽张量、$S\times S$ 的 128 MiB 等）在 $R$ 里，不在 $G$ 里。** 它们是前向算出来的 **激活**，不是参数，也不是梯度。

**Top-5 里第 4 项「FFN 参数矩阵，100 MiB」容易误会，单独说明：**

这里的「引用」**不是**「在显存里又存了一个指针/地址，指针本身占 100 MiB」。  
实际发生的是：做矩阵乘 $Y=XW_1$ 时，autograd 在 **自己的清单** 上记一笔——「反向算 $\partial W_1$ 时还要用到 $W_1$」。清单里登记的是 **$W_1$ 这张张量**（形状 $(d,d_{\mathrm{ff}})$，逻辑上 100 MiB），但 **GPU 并没有为此再 `cudaMalloc` 一块新的 100 MiB**；数据仍在账本 ① 里那块 **早就分配好的** $W_1$ 上，只是多了一条「反向时还要读它」的记录。

`saved_tensors_hooks` 的统计方式很「笨」：只要张量出现在 saved 列表里，就按 **元素个数 $\times$ 4 字节** 算体积，不管这块内存是不是新申请的。所以你会看到 **100 MiB 这个数字**，但它表示的是「**若按形状数元素，这张张量有多大**」，**不是**「又为参数多占了一块 100 MiB 显存」。  
同一物理显存：在参数账里算过一次（常驻），在 $R$ 的 hooks 统计里又按形状出现一次——**不要加两次**。真正 **新分配** 的、构成 $R$ 的大头，是 FFN 中间激活（680 MiB）和 $S\times S$（384 MiB）那些 **前向新算出来的激活**。

**一张图概括（第 16 层）：**

```
常驻（全程，不进 R/G）          前向多出来的（R）              反向多出来的（G）
─────────────────────          ─────────────────              ─────────────────
参数 W_q…W_o, W_1…W_3          FFN 中间激活      680 MiB       W_q.grad …   25 MiB ×4
  ≈400 MiB/层                    S×S 矩阵         384 MiB       W_1.grad …  100 MiB ×3
Adam m,v                         残差流激活       360 MiB       激活侧梯度   ~729 MiB
  ≈800 MiB/层                    hooks 对常驻 W   ≈100 MiB*     合计 G≈1129 MiB
  ↑ 38 GiB 地板（全网）            合计 R≈1566 MiB
  不算进 R，不算进 G              *按形状重复计数，非新分配
```

**读图：反向多出来的 $G\approx1129$ MiB 分两大赛道**

前向列 $R$ 存的是 **「当时的激活值」**（原料）；反向列 $G$ 写的是 **「梯度」**——loss 对每个量的敏感度。$G$ 不是一整块同质的显存，最自然的分法是：

| 赛道 | 图上写法 | 体积 | 一句话 |
|------|----------|-----:|--------|
| **参数梯度** | `W_q.grad` … `W_1.grad` … | **≈400 MiB** | 告诉 Optimizer：**每个参数矩阵该怎么改**；算完留在 `.grad` 里 |
| **激活侧梯度** | 「激活侧梯度 ~729 MiB」 | **≈729 MiB** | 告诉更前面的算子：**loss 对该层中间激活有多敏感**；算完要 **继续往前传**，传完往往就释放 |

实测 $G\approx1129\approx400+729$。下面分别说清楚。

---

**（一）参数梯度：$W_q.\mathrm{grad}$ 是什么？为什么 $\times4$ 和 $\times3$？**

参数梯度就是 **「这个参数矩阵该怎么改」** 的答案，形状与参数 **完全相同**，写入 `参数名.grad`，等 `optimizer.step()` 用。

第 16 层有两类参数矩阵，大小不同，所以 MiB 不同：

**Attention 四个 $d\times d$ 投影（$\times4$，各 25.0 MiB）**

| 符号 | 对应算子 | 前向在干什么 | 梯度 `.grad` 体积 |
|------|----------|--------------|------------------:|
| $W_q$ | `q_proj` | 把残差流投影成 Query | $2560^2\cdot4/1024^2=**25.0**$ MiB |
| $W_k$ | `k_proj` | 投影成 Key | 25.0 MiB |
| $W_v$ | `v_proj` | 投影成 Value | 25.0 MiB |
| $W_o$ | `output_proj` | 拼头后再投影回残差流 | 25.0 MiB |

四个矩阵 **形状一样**（都是 $d\times d=2560\times2560$），所以 **$\times4$，合计 $4\times25=100$ MiB**。

**SwiGLU 三个 $d\times d_{\mathrm{ff}}$ 矩阵（$\times3$，各 100.0 MiB）**

| 符号 | 对应算子 | 前向在干什么 | 梯度 `.grad` 体积 |
|------|----------|--------------|------------------:|
| $W_1$ | SwiGLU 升维支路 | $w_1(x)$，把 $(B,S,d)$ 拉到宽维 $(B,S,d_{\mathrm{ff}})$ | $2560\cdot10240\cdot4/1024^2=**100.0**$ MiB |
| $W_2$ | SwiGLU 降维 | 把宽维乘回 $(B,S,d)$ | 100.0 MiB |
| $W_3$ | gate 支路 | $w_3(x)$，与 $w_1$ 并行 | 100.0 MiB |

三个矩阵 **形状一样**（都是 $d\times d_{\mathrm{ff}}$），所以 **$\times3$，合计 $3\times100=300$ MiB**。

再加两个 RMSNorm 的 $\gamma$（各约 0.01 MiB），一层参数梯度 **$\approx400$ MiB**。  
图上写 `W_q.grad … 25 MiB ×4` 和 `W_1.grad … 100 MiB ×3`，就是这个意思：**不是**一个神秘的 $W_q$ 乘了 4，而是 **4 个** $d\times d$ 参数各有一份 25 MiB 的 `.grad`；**3 个** SwiGLU 参数各有一份 100 MiB 的 `.grad`。

---

**（二）激活侧梯度：是什么？为什么会有 ~729 MiB？**

你的理解方向 **对**：反向传播要一层层往前传，**每一个中间激活** 若还要参与前面的计算，就需要知道 **「loss 对它有多敏感」**——这就是激活侧梯度，常写成 $\partial L/\partial X$（$X$ 是某个中间激活）。

和另外两样东西务必分开：

| | 前向保存的激活（在 $R$ 里） | 参数的 `.grad` | **激活侧梯度** |
|--|---------------------------|----------------|----------------|
| **是什么** | 前向算出来的 **数值**（$w_1(x)$、scores…） | 参数 **该怎么改** | 中间激活 **该怎么往回传** |
| **何时产生** | 前向 | 反向 | 反向 |
| **典型命运** | 反向用完 **释放** | 留在 `.grad` 直到 `step` | 传给更前面的算子，**通常很快释放** |
| **例子** | 存下的 $w_1(x)$，80 MiB | $\partial W_1$，100 MiB | $\partial L/\partial X$（`ln2` 的输入），20 MiB |

**走一小段链，把概念钉死：**  
子层输出是 $Y=XW_o$（`output_proj`）。反向时，更靠近 loss 的一侧先传来 $\partial L/\partial Y$（「输出每个位置该怎么动，loss 才降」）。本步做两件事：

1. 算 $\partial W_o$ → 写入 `W_o.grad`（**参数梯度**，25 MiB，进上面 400 MiB 那赛道）；
2. 算 $\partial L/\partial X=(\partial L/\partial Y)\,W_o^{\top}$ → **激活侧梯度**（20 MiB，形状与输入 $X$ 相同），**交给更前面的 merge、attention…**

Attention、FFN 里每一个 matmul、softmax、silu 都是这样：**参数梯度留下改权重，激活侧梯度往前传**。所以反向不是只算 7 个 `.grad` 就结束，而是 **沿途写出许多张与中间激活同形的梯度张量**（20 MiB 的 $(B,S,d)$、80 MiB 的 $(B,S,d_{\mathrm{ff}})$、128 MiB 的 $(B,H,S,S)$……）。

**那 ~729 MiB 怎么理解？**  
它不是某一张固定形状的公式（不像 400 MiB 可以 $4\times25+3\times100$ 精确加出来），而是：

$$
\underbrace{1129}_{\text{实测 }G}
-\underbrace{400}_{\text{参数 }.grad}
\approx \underbrace{729}_{\text{激活侧梯度 + 反向临时缓冲}}.
$$

含义是：反向 **扫过整层** 的过程中，除了 400 MiB 的参数 `.grad`，还有大约 **729 MiB** 显存用在 **激活侧梯度**（以及少量尚未立刻释放的临时块）上。  
同一时刻 **不会** 把 B9 的 128 MiB、B17 的 80 MiB、各处 20 MiB **全部叠满**——算完一路就传给上一算子、或覆盖写回——但 hooks / $\Delta+R$ 估量的是 **整层反向期间出现过的梯度类显存规模**，所以仍远大于「只数 `.grad`」的 400 MiB。

和 $R$ 的对比（这也是 $G<R$、净降 437 MiB 的原因）：  
$R$ 里 FFN 中间激活 **存了 680 MiB 的前向数值**（多份、要留到反向）；反向时 **不会** 再写出 680 MiB 的「激活梯度」常驻——参数梯度只有 300 MiB 量级，激活侧梯度是 **分批、临时** 写的，峰值合计仍 **小于** 前向囤下的那 1.53 GiB 原料。

---

下面 §8.1 起，在这个分界前提下，把 $R$ 怎么涨到 1.53 GiB、$G$ 怎么落到 1.10 GiB、净少 437 MiB 顺着走完。

#### 8.1 前向跑过第 16 层：埋下 $R$

前向从 F1 走到 F18。每经过一个算子，autograd 可能把中间结果留下来，留给以后的反向用。  
**第 16 层全部前向算子跑完后**，这一层一共留下了 **46 个**保存张量，合计：

$$
R = 1565.7\ \text{MiB} \approx 1566\ \text{MiB} \approx 1.53\ \text{GiB}.
$$

这 **1.53 GiB 不是梯度**，是「前向为反传寄存的原料」。整网前向还没结束（第 17–32 层还要继续堆），但第 16 层的这份 $R$ **已经占着显存**，一直要等到反向扫回来才释放。

**这 1.53 GiB 里具体是什么？** 按体积从大到小（正文 Top-5 实测）：

| 排名 | 内容 | 体积 (MiB) | 怎么理解 |
|-----:|------|----------:|----------|
| 1 | FFN 中间激活（SwiGLU 的 $w_1x$、$w_3x$、`silu`、逐元乘等，多张 $(B,S,d_{\mathrm{ff}})$） | **680.0** | 单张 80 MiB；本层存了约 8.5 张 |
| 2 | Attention 的 $S\times S$（scores / mask 后 / 注意力权重，多张） | **384.0** | 单张 128 MiB；本层存了 3 张 |
| 3 | 残差流 / 隐状态（多处 $(B,S,d)$ 激活） | **360.0** | 单张 20 MiB；本层存了 18 张 |
| 4 | FFN 的 $W_1$ / $W_2$ / $W_3$ 之一（**常驻参数**；hooks 按形状记 100 MiB，**不是新分配的 100 MiB**，见 §8.0） | **100.0** | 逻辑体积 $(d,d_{\mathrm{ff}})$；物理上与账本 ① 同一块显存 |
| 5 | Attention 投影后的 $Q$ / $K$ / $V$（见下段说明；本层存了其中 **2 份**，各 20 MiB） | **40.0** | $2\times20$；前向为反传保存的激活，不是参数 |
| — | 其余 41 个小张量 | **≈2.0** | RMSNorm 统计、小张量视图等 |
| | **合计** | **≈1566** | |

前五项相加：$680+384+360+100+40=1564$ MiB，与总量 1566 MiB 差约 2 MiB，落在「其余小项」里。

**第 4、5 项再展开一句：**

- **第 4 项（100 MiB）**：SwiGLU 做 $Y=XW_1$ 时，autograd 把 **已经在 GPU 上的** $W_1$ 登记进 saved 列表。hooks 看到形状 $(2560,10240)$，就记 **100 MiB**——这是 **按元素个数换算的统计数字**，不是又多占了一块显存（详见 §8.0）。
- **第 5 项（40 MiB）**：Attention 里先把输入投影成 Query、Key、Value 三路（`q_proj` / `k_proj` / `v_proj`），再 **按 32 个头拆开**，每路的形状是「batch $\times$ 头数 $\times$ 序列长 $\times$ 每头维度」即 $(4,32,512,80)$，逻辑体积与一张残差流同为 **20 MiB**。反向穿过 attention 时，需要用到其中 **若干份** 投影结果（例如算 $QK^\top$ 要 $Q$ 和 $K$，算 注意力权重$\times V$ 要 $V$）；本层 hooks 统计到 **2 张** 这样的张量被保存，故 $2\times20=40$ MiB。它们是 **前向新算出来、为反传留下的激活**，不是 $W_q,W_k,W_v$ 那些参数矩阵本身。

此时若只盯着第 16 层：显存里多了一块 **约 1.53 GiB** 的「保存张量账」。

#### 8.2 整网前向结束：第 16 层的 $R$ 仍活着

32 层全部前向跑完后，**每一层**都有自己的 $R$ 叠在显存里（没有 checkpoint 就不会提前释放）。  
第 16 层那份 1566 MiB **一直在**；它不会因为在第 17 层又算了新东西就自动消失。

整步前向峰值（正文 (a) train 曲线）大约在 **65.5 GiB** 附近——其中包含：参数 + Adam（约 38 GiB 地板）+ **32 层各自的 $R$ 叠加** + loss 附近临时量。  
单看一层：$R\approx1.5$ GiB 只是整步显存里的一小块；32 层粗算 $32\times1.5\approx48$ GiB 量级，与「激活远大于参数地板」的观测一致。

#### 8.3 反向扫到第 16 层：同时发生两件事

反向从第 32 层往第 1 层走。轮到第 16 层时：

1. **读**前向留下的 $R$（1566 MiB），按 §4 的 B18→B1 逐步算梯度；
2. **写**梯度张量 $G$（参数 `.grad` + 激活侧梯度）；
3. **释放**已用完的保存张量（$R$ 逐项归还显存）。

`memory_allocated` 在这一小段里只能看到 **净变化** $\Delta$。实测第 16 层（各层接近）：

$$
\Delta \approx -437\ \text{MiB}.
$$

负号表示：**撤走的比新写的多**，显存净下降 437 MiB。

由 $\Delta \approx G - R$ 反推这一层写出的梯度总量：

$$
G \approx \Delta + R \approx (-437) + 1566 = 1129\ \text{MiB} \approx 1.10\ \text{GiB}.
$$

所以：**前向存了约 1.53 GiB 原料，反向写出约 1.10 GiB 梯度，净少占约 437 MiB。**

#### 8.4 写出的 $G\approx1.10$ GiB 里有什么？

**（1）参数梯度（可精确加总）**

第 16 层所有可学习参数各有一份 `.grad`，形状与参数相同。一层内：

$$
\begin{aligned}
&4\times 25.0 && (W_q,W_k,W_v,W_o:\ d\times d) \\
&+\ 3\times 100.0 && (W_1,W_2,W_3:\ d\times d_{\mathrm{ff}}) \\
&+\ 2\times 0.01 && (\text{两个 RMSNorm 的 }\gamma) \\
=\ &400.02\ \text{MiB}.
\end{aligned}
$$

这是 $G$ 里 **一定能对上公式** 的部分。

**（2）激活侧梯度（实测 $G$ 减去上面的 400 MiB）**

$$
1129 - 400 \approx 729\ \text{MiB}.
$$

这是反向过程中 **暂时写出、用于继续往前传** 的激活梯度，以及部分尚未立刻释放的临时缓冲。它们不是「又多存了一份前向激活」，而是 **梯度**。

按 §4 反传表，这一层里单步新增较大的几项（同一时刻不会全部叠满，但量级如下）：

| 反向步骤 | 本步新写（MiB） | 写出的是什么 |
|----------|----------------:|--------------|
| B17 $w_2$ 反传 | $100.0+80.0=180.0$ | $\partial W_2$（参数）+ $\partial Z$（激活） |
| B9 注意力权重 $\times V$ | $20.0+128.0=148.0$ | $\partial V$ + $\partial$(注意力权重) |
| B10b `output_proj` | $25.0+20.0=45.0$ | $\partial W_o$ + $\partial L/\partial X$ |
| B6 $QK^{\top}$ | $20.0+20.0=40.0$ | $\partial Q$ + $\partial K$ |
| B15 `silu` | $80.0$ | $\partial u$ |
| B14/B13 $w_3,w_1$ | 各 $100.0$ | $\partial W_3$、$\partial W_1$（参数） |
| B2–B4 Q/K/V 投影 | 各 $25.0$ | $\partial W_q$、$\partial W_k$、$\partial W_v$ |

参数 `.grad` 合计 **400 MiB** 会 **留在** `.grad` 里直到 `optimizer.step()`；激活侧梯度（如 128 MiB 的 $S\times S$ 块、80 MiB 的 FFN 宽形）算完往往 **释放或覆盖**，但反向扫过整层期间它们曾占用显存，计入 $G$ 的估量。

于是：**$G\approx1129$ MiB $\approx$ 400 MiB（参数梯度，常驻到 step）$+$ ~729 MiB（反向过程中的激活梯度与临时块）。**

#### 8.5 把数字顺着排：$1.53$ GiB 到 $1.10$ GiB 差在哪儿

下面只列 **第 16 层自己的账**，不牵涉其它层：

| 阶段 | 显存事件 | 体积 (MiB) | 累计含义 |
|------|----------|----------:|----------|
| 前向结束 | 第 16 层保存张量 $R$ 入账 | **+1566** | 这一层占着 1.53 GiB「原料」 |
| 反向开始扫过 | 读 $R$，逐步释放 | **−1566** | 原料用完归还 |
| 反向同时 | 写梯度 $G$ | **+1129** | 写出 1.10 GiB「梯度」 |
| **净效果** | $\Delta = G - R$ | **−437** | 这一层净少占 437 MiB |

**437 MiB 从哪来？** 把 $R$ 与 $G$ 按内容对一下：

| $R$ 里占大头（前向保存） | 体积 | 反向写出的大头（$G$ 侧） | 体积 |
|--------------------------|-----:|--------------------------|-----:|
| FFN 中间激活 | 680 | 三个 FFN 参数梯度 $\partial W_1,\partial W_2,\partial W_3$ | $3\times100=300$ |
| $S\times S$ attention | 384 | 注意力路径上的激活梯度（临时，单块最大 128） | $\lesssim 128$ 级 |
| 残差流 $(B,S,d)$ 多处 | 360 | 各算子激活梯度（单块 20，逐步释放） | 分散、不同时峰值 |
| 参数视图等 | 140 | Q/K/V/O 四个 $d\times d$ 参数梯度 | $4\times25=100$ |
| 小项 | $\sim2$ | RMSNorm $\gamma$ 等 | $\sim0$ |

前向为了反传 **按原样保存** 了大块激活（尤其 FFN 680 MiB、$S\times S$ 384 MiB）；反向算完后，**不必**也 **不会** 用同等体积的「激活梯度」替换它们——参数梯度总共才 400 MiB，激活梯度也是算完就传、就释，峰值远小于前向存下的那份 $R$。

所以同一层上：**$R$（1566）$>$ $G$（1129）**，差额 **437 MiB**。这就是反向经过一层时显存台阶 **往下走** 的直接原因。

#### 8.6 时间轴（一层粒度）

```
前向 F1…F18（第16层）
  └─ 显存 +1566 MiB（R：保存张量，一直占着）

… 第17–32层继续前向，整网前向峰值 …

反向从第32层往回 …
  └─ 轮到第16层：
       读 R（1566 MiB）→ 写 G（1129 MiB）→ 释放 R
       净变化 Δ ≈ −437 MiB

… 继续往第15层 …
```

**三个数记住即可：**

- **$R\approx1566$ MiB $\approx1.53$ GiB** — 前向存的（实测，Top-5 见 §8.1）  
- **$G\approx1129$ MiB $\approx1.10$ GiB** — 反向写的（由 $\Delta+R$ 估得）  
- **$|R-G|\approx437$ MiB** — 反向扫过这一层时，这层对显存的净贡献从「占 1.53 GiB 原料」变成「占 1.10 GiB 梯度」，**少占 437 MiB**

参数梯度公式下界 **400 MiB** 只是 $G$ 里「参数 `.grad`」那一块；**$G$ 实测 1129 MiB**，中间大约 **729 MiB** 是反向过程中的激活梯度与临时缓冲。  
**$R$ 实测 1566 MiB** 则主要是 FFN 中间激活（680）和 $S\times S$（384）等前向保存物——体积大，但反向不会等量换成梯度留回去，所以 $R>G$，$\Delta<0$。
