# CS336 §6：Optimizer State Sharding（讲义约第 37 页，15 分）

> 读完应能直接开工：真实机制 → 一步里谁存什么 → 通信传什么 → 接口与文件落点。  
> 不摊开完整实现代码。Accounting / 2 卡实验有卡再做。

---

## 1. 这题升级的是哪一块

| 已有 | 本题 |
|------|------|
| DDP（含 Overlap）：管 **梯度** 何时/如何跨卡平均 | 不动这块也能单独测本题 |
| PyTorch 自带 `AdamW`：每卡对 **全部参数** 建 OPT 状态并 `step` | 换成 **ShardedOptimizer**：OPT 状态约切成 `1/world_size` |

可切的三样（ZeRO 谱系）：**Parameter / Gradient / OPT**。  
本题主攻 **OPT**。Parameter 整存、按层切开是后话（FSDP）；Gradient 分片是 ZeRO-2，本题不要求。

**和切 batch 无关。** 测试里甚至每卡同一份输入。做题时先把「data parallel 切数据」从脑子里拿开。

---

## 2. 全局调用图：test → adapter → 你的模块（函数级）

三层只干这些事：

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  tests/test_sharded_optimizer.py                                         │
│                                                                          │
│  test_sharded_optimizer(model_class)                                     │
│       │                                                                  │
│       │  mp.spawn(_test_sharded_optimizer, nprocs=world_size=2)          │
│       ▼                                                                  │
│  _test_sharded_optimizer(rank, world_size, model_class)                  │
│       │                                                                  │
│       ├─ _setup_process_group(...)     # 建进程组；之后才有 rank/world  │
│       │                                                                  │
│       ├─ non_sharded_model = ToyModel()                                  │
│       │  non_sharded_optimizer = AdamW(all params, lr=...)               │
│       │       ▲ 对照基线：完整 OPT，不切分片                             │
│       │                                                                  │
│       ├─ sharded_model = deepcopy(non_sharded_model)                     │
│       │                                                                  │
│       │  sharded_optimizer = get_sharded_optimizer(                      │
│       │        sharded_model.parameters(),  ─────────┐                   │
│       │        optimizer_cls=AdamW,                  │ 传入：全参数迭代器│
│       │        lr=..., betas=..., ...)               │ + 优化器类 + 超参 │
│       │                                              ▼                   │
│       │                         ┌────────────────────────────────────┐   │
│       │                         │ tests/adapters.py                  │   │
│       │                         │ get_sharded_optimizer(params,      │   │
│       │                         │     optimizer_cls, **kwargs)       │   │
│       │                         │   → return ShardedOptimizer(...)   │   │
│       │                         │     （薄包装，几乎只做 import）      │   │
│       │                         └──────────────┬─────────────────────┘   │
│       │                                        ▼                         │
│       │                         ┌────────────────────────────────────┐   │
│       │                         │ cs336_systems/distributed/         │   │
│       │                         │   sharded_optimizer.py             │   │
│       │                         │                                    │   │
│       │                         │ __init__(params, optimizer_cls,    │   │
│       │                         │          **kwargs)                 │   │
│       │                         │   ├─ super().__init__(...)         │   │
│       │                         │   ├─ add_param_group 路径里分片    │   │
│       │                         │   │    本 rank 只留下 ≈1/W 参数    │   │
│       │                         │   └─ self._opt = optimizer_cls(    │   │
│       │                         │          本分片参数, **kwargs)     │   │
│       │                         │        ↑ 只有这里有 Adam m/v       │   │
│       │                         └────────────────────────────────────┘   │
│       │                                                                  │
│       │  for _ in range(10):                                             │
│       │      │                                                           │
│       │      ├─ sharded_optimizer.zero_grad()                            │
│       │      │      └─（通常清 sharded_model 上参数的 .grad）            │
│       │      │                                                           │
│       │      ├─ logits = sharded_model(x)     # 整模 forward             │
│       │      ├─ loss = ...                                               │
│       │      ├─ loss.backward()               # 整模 backward            │
│       │      │      └─ 各 param.grad 写满（完整梯度）                    │
│       │      │                                                           │
│       │      └─ sharded_optimizer.step()                                 │
│       │             │                                                    │
│       │             ▼                                                    │
│       │      ┌─ 你的 ShardedOptimizer.step() ─────────────────────────┐  │
│       │      │  (1) self._opt.step()                                   │  │
│       │      │        只用本分片 param 的 .grad                        │  │
│       │      │        只改本分片 param.data                            │  │
│       │      │        只动本分片 Adam 状态                             │  │
│       │      │  (2) broadcast 本分片更新后的权重 (src=本 rank)         │  │
│       │      │        同时接收其他 rank 的分片                         │  │
│       │      │        → sharded_model 上完整权重再次对齐               │  │
│       │      └─────────────────────────────────────────────────────────┘  │
│       │                                                                  │
│       └─ assert sharded_model.parameters ≈ non_sharded_model.parameters  │
└──────────────────────────────────────────────────────────────────────────┘
```

**数据在接口上流动的只有这些：**

```text
构造期:
  sharded_model.parameters() ──► get_sharded_optimizer ──► ShardedOptimizer.__init__
       │                                                      │
       │                                                      ├─ 分片归属
       │                                                      └─ AdamW(本分片, lr/betas/...)
       └─ 参数张量仍活在 sharded_model 上（完整一份）

一步训练:
  x,y ──► sharded_model.forward ──► loss.backward ──► 各 param.grad
                                                      │
  sharded_optimizer.step() ◄── 读本分片 .grad，写本分片 .data
         │
         └─ broadcast(.data 分片) ──► 其他 rank 的同名参数 .data
```

对照基线在同进程里并行跑：`non_sharded_optimizer.step()` 更新**全部**参数且**无** broadcast；最后比的是两边 **model 权重**是否一致。

---

## 3. 真实运行机制（每卡内部）

每张卡上发生的是：

```text
整份 Model（全部 Parameter）
    → 完整 Forward
    → 完整 Backward
    → 几乎每个可训练 Parameter 上都有一份 .grad   ← 自然结果，不是「多挂」出来的
    → 本地 Optimizer 只「认领」约 1/4 参数：
         只为它们存 Adam 状态（m/v 等）
         step 时只用它们的 .grad 更新它们的权重
    → Broadcast：把我刚更新的那 1/4 权重发给别人；
                 同时收齐别人更新的 3/4
    → 下一轮开始时，每卡权重再次完整且一致
```

一句话：

> **算**用整网；**OPT 状态与更新权**切开；**用 broadcast 把权重拼回一致。**

不是流水线：不是「卡0 管左 1/4 层、卡3 管右 1/4 层」。每卡都跑整网。

---

## 4. 为什么 Gradient「看起来是完整的」

Backward 按计算图从 loss 往回走，会给 **参与了前向的可训练参数** 写入 `.grad`。  
本地 `step` 就算只更新 1/4 参数，**也不会让 autograd 自动跳过另外 3/4 的梯度计算**。

因此：

- 完整 `.grad` = 完整 backward 的副产物。  
- ShardedOptimizer 只是 **step 时不消费** 非本分片的 `.grad`（不建状态、不改那些权重）。  
- 非本分片权重的新值来自 **别的 rank 的 step + broadcast**，不是来自「我没算它们的 grad」。

（若做成 ZeRO-2，才会在通信/存储上进一步处理梯度分片；本题不必。）

---

## 5. 每张卡存什么（4 卡示意）

| | 每张卡 |
|--|--------|
| Parameter（计算用） | **完整** |
| `.grad`（backward 后） | **完整**（直到你手动清掉） |
| Adam 等 OPT 状态 | **仅本分片** ≈ 1/4 |
| 本步本地改动的权重 | **仅本分片**；随后靠 broadcast 对齐 |

省显存的主头：**OPT 状态**（Adam 大约再占两倍参数量级），不是「少存 3/4 参数」。

---

## 6. 分片规则（实现时要对准的语义）

切的是：**哪些 Parameter 交给本 rank 的底层 `optimizer_cls` 实例。**

建议流程：

1. 收集待优化参数（支持 list 或 param group dict）；**tied weights 按 `id` 去重**。  
2. 按 `rank` / `world_size` 划分（如 `i % world_size == rank`，或连续切块）。  
3. 只有本分片进入本地 `AdamW(...)`；其余参数本 rank 不建 OPT 状态。

分片逻辑应能落在 **`add_param_group`**：基类构造会调它，训练中加 group 也会调它。

---

## 7. 一步更新 + 通信（broadcast，不是 all-reduce）

```text
zero_grad → forward → backward
    →（若与 DDP 联用：先梯度 all-reduce；本题单测可不做）
    → sharded_optimizer.step():
         (a) 底层 optimizer 只更新本分片参数
         (b) 对本分片每个（或打包后的）参数: broadcast(src=本 rank)
         (c) 同时作为接收方，收齐其他 src 的分片
```

### 7.1 和「0～15」例子对齐的粒度

更新刚结束、通信前（示意：每卡负责 4 个数）：

```text
卡0 刚写好的分片: [0, 1, 2, 3]
卡1 刚写好的分片: [4, 5, 6, 7]
卡2 刚写好的分片: [8, 9, 10, 11]
卡3 刚写好的分片: [12, 13, 14, 15]
```

此时每卡上「别人的分片」还是旧值。然后：

```text
broadcast(卡0 的分片, src=0) → 覆盖所有人对应位置
broadcast(卡1 的分片, src=1) → …
…
```

结束后四人完整参数一致。  
**传的是更新后的权重，不是 batch，也不是「把梯度平均成一份再当权重」。**

| | 梯度 all-reduce（DDP） | 参数 broadcast（本题） |
|--|----------------------|------------------------|
| 对象 | `.grad` | 刚被本地 step 改过的 **weight** |
| 语义 | 各卡贡献合成（常平均） | `src` 权威副本覆盖他人 |

### 7.2 「我这台机器上的数据怎么到别人那」

若指 **batch**：不到别人那去训；各卡自有（或测试相同）。  
若指 **我更新的 1/4 权重**：`broadcast(src=我)` 发出去；别人的 3/4 权重由他们 `broadcast` 给我。

---

## 8. 全局接线（仓库）

```text
tests/test_sharded_optimizer.py
  get_sharded_optimizer(model.parameters(), AdamW, lr=...)
  循环 step 后：权重应 ≈ 普通非分片 AdamW

tests/adapters.py
  get_sharded_optimizer(...) → 返回你的类

建议文件:
  cs336_systems/distributed/sharded_optimizer.py
```

测试骨架：

```text
model 整份 + sharded opt
zero_grad → forward → loss → backward → opt.step()  ×10
assert 与「完整 AdamW」权重接近
```

验收：

```bash
cd /root/.dev/ml-sys/cs336/assignment2-systems && .venv/bin/python -m pytest tests/test_sharded_optimizer.py -v
```

---

## 9. 接口方向（讲义）

类继承 `torch.optim.Optimizer`。

**`__init__(self, params, optimizer_cls, **kwargs)`**  
- `params`：全部参数或 param groups。  
- `optimizer_cls`：如 `AdamW`（类）。  
- `kwargs` → 底层构造。  
- 必须 `super().__init__(...)`（讲义要求）。  
- 内部持有：只含本分片的 `optimizer_cls(...)`。

**`step(self, closure=None, **kwargs)`**  
- 先底层 `step`；再对本分片参数做 `broadcast`。

**`add_param_group(self, param_group)`**  
- 构造与中途加组都走这里；在此完成分片归属。

---

## 10. 实现清单

1. `Optimizer` 子类 + `super().__init__` 与 `param_groups` 约定兼容。  
2. `add_param_group`：去重、按 rank 切、子集交给 `optimizer_cls`。  
3. `step`：本地更新 → `broadcast(src=rank)`。  
4. tied weights 只分给一个 rank。  
5. `zero_grad` 行为与测试一致（常随基类 / 底层 opt）。  
6. adapter 接线；pytest 多跑几次。

---

## 11. 总纲

> 每卡整网算完 → 梯度自然齐全 → OPT 只养 1/4 → step 只改 1/4 → broadcast 把权重对齐。  
> 省的是 Adam 状态；不是把模型切成四段流水线。
