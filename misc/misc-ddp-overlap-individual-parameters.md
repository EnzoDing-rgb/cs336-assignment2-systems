# CS336 §5.3.2：Overlapping DDP（逐参数、与 backward 重叠）

> 对应讲义约第 35–36 页：`ddp_overlap_individual_parameters`（以及同页的 benchmarking 题，**需要 2 卡，先不做实验**）。  
> 本文按「认知顺序」写：你先要懂什么 → 这题到底要你交什么 → 实现时脑子里该有的时间线 → 必备 API → 和你已有代码的关系 → 怎么验收。

---

## 0. 你现在站在哪（先对齐已有工作）

你已经做过两件事：

| 实现 | 通信怎么做 | 通信发生在什么时候 |
|------|------------|--------------------|
| `NaiveDDP` | **每个参数**各自 `all_reduce` 一次 | **整个** `loss.backward()` **结束之后**，在 `finish_gradient_synchronization()` 里 |
| `FlattenDDP` | 先把所有梯度 **flatten** 成一条，**只** `all_reduce` 一次，再拆回 | 同样是 **backward 全部结束之后** |

两者的共同点：**通信都排在 backward 后面**。  
时间线长这样：

```text
forward → backward（整段算完梯度）→【通信】→ optimizer.step
              ↑                         ↑
         这段时间 GPU 在算           这段时间 GPU/网络在传
         通信在干等                    计算在干等
```

讲义 §5.3 说：最小实现有两个痛点——

1. **通信次数太多**（每个参数一次）→ 用 flatten 打包缓解（你已做的 5.3.1）。  
2. **即使打包了，通信时间仍然整段加在关键路径上** → 这就是 **5.3.2** 要解决的：让通信和 backward **叠在一起**。

---

## 1. 核心直觉：backward 不是「一瞬间算出所有梯度」

反向传播是 **从 loss 往输入一层一层推** 的：

```text
loss
  → 先算出「靠近输出」的那些参数的梯度
  → 再往前算中间层
  → 最后才算靠近输入的层
```

所以：**不是**等所有 `.grad` 都齐了才有东西可传；  
而是：**某个参数的梯度一旦算完，就可以立刻开始 all-reduce**，与此同时 PyTorch 还在算别的参数的梯度。

理想时间线变成：

```text
forward → backward 开始
            ├─ 参数 A 的 grad 好了 ──► 立刻开始传 A（异步）
            ├─ 同时继续算参数 B 的 grad
            ├─ 参数 B 的 grad 好了 ──► 立刻开始传 B（异步）
            └─ …
         backward 的 Python 调用返回
         → finish_gradient_synchronization()：等到所有「传梯度」真正做完
         → optimizer.step()
```

「重叠（overlap）」指的就是：**算后面的梯度** 和 **传前面已经算好的梯度** 在时间上交叠，从而缩短「从 backward 开始到可以 step」的墙钟时间。

---

## 2. 这题要你干什么（Deliverable）

### 2.1 实现题：`ddp_overlap_individual_parameters`（5 分）

写一个 **DDP 容器类**（名字自定，例如 `OverlapDDP`），包装任意 `nn.Module`，负责：

1. **构造时**：把 rank 0 的参数（以及通常还有 buffer）**broadcast** 到所有 rank（和 Naive 一样，保证初始权重一致）。  
2. **backward 过程中**：每当某个参数的梯度就绪，就对该参数的 `.grad` 发起 **异步** `all_reduce`（仍然是 **逐参数**，不是 flatten 成一条）。  
3. **提供** `finish_gradient_synchronization()`：在 `optimizer.step()` 之前调用，**等待**所有异步通信完成（并完成「求和 → 平均」若你放在通信后处理）。

讲义推荐的对外接口：

```text
__init__(module)
forward(*inputs, **kwargs)          # 转给内部 module
finish_gradient_synchronization()   # 等异步通信结束
```

讲义给的使用方式：

```python
model = ToyModel().to(device)
ddp_model = DDP(model)          # 你的容器

for _ in range(train_steps):
    x, y = get_batch()
    logits = ddp_model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()
```

接线到作业测试：

- 实现 `adapters.get_ddp` → 返回你的 overlap 容器实例。  
- `adapters.ddp_on_after_backward`：**可选**；常见写法仍是调用 `finish_gradient_synchronization()`（测试在 backward 和 `optimizer.step` 之间会调它）。  
- 验收：`uv run pytest tests/test_ddp.py`（讲义建议多跑几次，例如 5 次，抓竞态）。

> 注意：`get_ddp` 的 docstring 写的就是「backward 里异步、逐参数通信」。  
> 你现在 adapter 还指着 `NaiveDDP`；做完 overlap 实现后，**应改成返回 overlap 版本**（Naive / Flatten 留给 benchmark 对比，不必删）。

### 2.2 Benchmarking 题（同页，1 分）——**需要 1 node × 2 GPUs**

- (a) 用 xl 模型测 overlap 版每步时间，并和 Naive（逐参数、backward 后同步）、Flatten（一条 all_reduce）对比。  
- (b) 用 Nsight 各截一张：一张看出「通信没和 backward 重叠」，一张看出「有重叠」。

**当前机器若只有 1 卡：实现与 pytest（gloo 双进程）可以先做；2 卡计时 / Nsight 明确延后。**

---

## 3. 你必须先具备的知识（按学习顺序）

下面按「不懂就写不出」的依赖顺序排列。

### 3.1 复习：DDP 在数学上要保证什么

- 每张卡一份完整模型；数据按 rank 切片。  
- 各卡算出「本地数据」上的梯度后，要对齐成 **全局平均梯度**，再 `optimizer.step()`。  
- 初始权重必须一致 → 构造时 `broadcast`。  
- 平均常见写法：`all_reduce(..., SUM)` 再 `div_(world_size)`，或 `ReduceOp.AVG`。

这些和 Naive / Flatten **完全一样**；变的只是 **何时发起通信** 以及 **是否 async**。

### 3.2 Backward 增量产生梯度

你需要建立这个信念：

> `loss.backward()` 内部是按计算图反向遍历的；  
> **每个参数的 `.grad` 是在不同时刻才写好的**，不是最后一刻同时出现。

因此「梯度就绪 → 立刻通信」在机制上是可行的。

### 3.3 Backward hook：梯度一攒好就回调你

讲义点名 API：

**`Tensor.register_post_accumulate_grad_hook(fn)`**

含义（人话）：

- 在 autograd 给这个参数 **累加完本次的梯度** 之后，自动调用你的 `fn`。  
- 于是你可以在 hook 里对 `param.grad` 发起 `all_reduce`，而不必自己手写「backward 走到哪一层了」。

典型用法思路（伪代码，实现时再抠细节）：

```python
def make_hook(param):
    def hook(param):  # 签名以官方文档为准
        # 此时该 param 的 grad 已经为本 step 准备好（在 post-accumulate 语义下）
        handle = dist.all_reduce(param.grad, async_op=True, ...)
        handles.append(handle)
        # 平均可以在 wait 之后做，或用 AVG / 在 hook 里谨慎处理
    return hook

for p in module.parameters():
    if p.requires_grad:
        p.register_post_accumulate_grad_hook(make_hook(p))
```

你需要自己搞清楚的细节（实现时查文档 / 试）：

- hook 何时注册（通常在 `__init__`）。  
- **tied weights**（同一 `Parameter` 多个引用）：只注册一次 / 只通信一次（和 Naive 用 `id` 去重同一思想）。  
- `requires_grad=False` 的参数：不要注册或不通信。  
- 平均（`/ world_size`）放在 hook 里还是 `wait()` 之后：两种都能做对，但要保证 **所有 rank 一致**，且 **step 之前平均已完成**。

官方文档：  
https://pytorch.org/docs/stable/generated/torch.Tensor.register_post_accumulate_grad_hook.html

### 3.4 异步集合通信：`async_op=True` 与 `handle.wait()`

讲义把同步 / 异步对比写得很清楚，必须读懂这段：

**`async_op=False`（你 Naive / Flatten 现在用的）：**

- 调用返回时：集体操作至少已经 **入队到 GPU**（对 NCCL/CUDA 而言）。  
- **不等于** 通信已经在硬件上跑完；但后续依赖该张量的算子一般会按正确依赖排队。  
- 对你写 Naive 时：逻辑简单——调用返回后可以立刻 `div_`，再 `step`（测试路径里还有 `ddp_on_after_backward`）。

**`async_op=True`（overlap 要用的）：**

- 调用立刻返回一个 **handle**。  
- 返回时：**甚至不保证** 通信已经入队，更不保证完成。  
- 在依赖这些梯度之前（尤其是 `optimizer.step()` 之前），必须对每个 handle 调用 **`handle.wait()`**。

讲义示例：

```python
handles = []
for tensor in tensors:
    handle = dist.all_reduce(tensor, async_op=True)
    handles.append(handle)

# ... 中间可以干别的、不依赖这些 all_reduce 结果的事 ...

for handle in handles:
    handle.wait()
handles.clear()
```

对应到 DDP 容器：

- hook 里：`async_op=True`，把 handle 存进列表。  
- `finish_gradient_synchronization()`：对所有 handle `wait()`，清列表；必要时再做平均。

### 3.5 为什么「异步 + hook」才能重叠

关键对比：

| | Naive（同步、backward 后） | Overlap（异步、hook 里） |
|--|---------------------------|---------------------------|
| 发起 all_reduce 的时机 | backward 完全结束 | 每个参数 grad 一好就发 |
| `all_reduce` 是否卡住 backward | 不参与 backward；backward 早结束了，通信在后面串行 | `async_op=True` 不阻塞，backward 继续算别的层 |
| `finish_...` 的职责 | **亲自做** 全部 all_reduce（或你已做完） | **等待** 已经发起的异步通信结束 |

重叠发生在：`loss.backward()` **还在跑** 的那段墙钟时间里，通信已经在飞。

### 3.6 和 Flatten 的关系（避免概念打架）

| | Flatten（5.3.1） | Overlap 逐参数（5.3.2） |
|--|-----------------|-------------------------|
| 优化目标 | 减少 **通信调用次数** | 让通信与 **backward 计算** 重叠 |
| 何时通信 | backward **之后** | backward **之中** |
| 每次传什么 | 一条拼好的大向量 | **单个**参数的 grad |
| 是否矛盾 | 不矛盾；是两条不同改进路线 | 讲义后面还有「bucket / 打包 + overlap」等更强组合，本题先做 **逐参数 overlap** |

本题明确要求：**individual parameter tensors**（逐参数），不要在这题里改成只 flatten 一次。

---

## 4. 推荐实现时的思维清单（仍按顺序）

1. **抄 Naive 的外壳**：`nn.Module` 子类、`self.module`、`forward` 转发、`broadcast`。  
2. **在 `__init__` 里注册 hook**（对需要梯度的、去重后的参数）。  
3. **hook 里**：`all_reduce(..., async_op=True)`，保存 handle；想清楚平均何时做。  
4. **`finish_gradient_synchronization`**：`wait` 所有 handle；补齐平均；清空 handle 列表，避免跨 step 残留。  
5. **每个 training step**：通常还要 `optimizer.zero_grad()`（测试里有）；新 step 的 hook 会再次往 handle 列表里塞。  
6. **改 `get_ddp`** 返回 overlap 类；保留 `ddp_on_after_backward` → `finish_...`。  
7. **先跑通** `pytest tests/test_ddp.py`（本机 gloo、world_size=2 即可，**不必 2 张真 GPU**）。  
8. **2 卡 xl + Nsight**：有多卡再做 benchmarking 小题。

---

## 5. 和测试如何对上（避免实现完对不上）

`tests/test_ddp.py` 大致期望：

1. `get_ddp(model)` 返回的对象能 `.parameters()` / 有 `.module`（`validate_ddp_net_equivalence` 用 `net.module.state_dict()`）。  
2. 构造后非 0 号 rank 的权重应已被 broadcast 成与 rank 0 一致。  
3. 训练循环：`backward` → `ddp_on_after_backward` → `optimizer.step()`，最终与「单进程看全数据」的 baseline 参数对齐。

因此：只要你在 `finish_...`（由 adapter 调用）时保证 **梯度已平均且通信已完成**，测试不关心你是「backward 后同步 all_reduce」还是「hook 异步 + wait」——但 **本题评分意图** 是后者。

---

## 6. 本题明确不要求你现在做的事

- 不要求先改 Flatten 的算法。  
- 不要求本机 2 卡跑 xl（那是 benchmarking 小题）。  
- 不要求 Nsight（同题 (b)）。  
- 不要求上 FSDP / optimizer sharding（更后面章节）。

---

## 7. 一句话总纲

> **Naive**：backward 算完 → 再同步传梯度。  
> **Flatten**：backward 算完 → 打包成一条再传（少次数）。  
> **Overlap（本题）**：backward **边算边传**（逐参数、异步），`finish_...` 只负责 **等传完** 再 `step`。

读完本文后，下一步才是：新建 overlap 类骨架（对齐 Naive 的 broadcast/forward），再填 hook + async + `wait`。
