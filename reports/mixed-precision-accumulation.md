# Mixed-Precision Accumulation

Assignment 2 `mixed_precision_accumulation`。脚本：`cs336_systems/mixed_precision/mixed_precision_accumulation.py`。
运行：`uv run --no-sync python -m cs336_systems.mixed_precision.mixed_precision_accumulation`。

理想精确值：\(1000 \times 0.01 = 10\)。

## 实测输出

| case | 做法 | 打印结果 |
|---|---|---|
| 1 | FP32 累加器 `+=` FP32 加数 | `10.0001` |
| 2 | FP16 累加器 `+=` FP16 加数 | `9.9531`（fp16） |
| 3 | FP32 累加器 `+=` FP16 加数 | `10.0021` |
| 4 | FP32 累加器 `+=`（先把 FP16 转成 FP32） | `10.0021` |

## 人话解读

- **Case 1**：全程 FP32。浮点本身就有舍入，所以不是完美的 `10.0`，而是 `10.0001`，误差极小，可以当「准」的基准。
- **Case 2**：累加器也是 FP16。FP16 尾数位少，一边加一边把误差写回低精度寄存器；加到后面，`0.01` 相对当前总和已经小到容易被「吃掉」或舍得很狠，所以掉到 `9.9531`，**明显偏了**。这就是讲义要你看见的：低精度累加会伤精度。
- **Case 3 / 4**：累加器保持 FP32。Case 3 里 `+=` 会把 FP16 加数提升后再加；Case 4 是你手动 `.type(float32)` 再加——效果一样，都得到 `10.0021`。加数 `0.01` 本身用 FP16 表示就不完美，所以和 Case 1 略有差别，但**远好于 Case 2**。

一句话：**张量可以是低精度，但求和/累加最好放在更高精度里做**；否则误差会在循环里被反复放大、写死。

## Deliverable（2–3 句）

纯 FP32 累加得到约 `10.0001`，接近真值 10；把累加器也改成 FP16 后结果变成约 `9.9531`，误差明显变大。若累加器保持 FP32、只把每次加数用 FP16（或先 cast 回 FP32 再加），结果约 `10.0021`，精度接近 FP32 累加。这说明混合精度里应把 reduction/accumulation 留在较高精度，即使参与运算的张量本身是低精度。
