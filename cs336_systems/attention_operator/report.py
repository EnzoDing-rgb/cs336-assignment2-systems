"""Generate benchmarking-scaled-dot-product-attention.md from sweep results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from cs336_systems.attention_operator.benchmark import BenchmarkResult, DEFAULT_BATCH_SIZE

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
ARTIFACTS_PATH = REPO_ROOT / "artifacts" / "attention_operator" / "results.json"

B = DEFAULT_BATCH_SIZE
BYTES_PER_ELEM = 4


def _img(name: str, width: int = 620) -> str:
    path = FIGURES_DIR / name
    return f'<img src="{path}" alt="{name}" width="{width}" />'


def _theoretical_saved_gib(s: int, batch: int = B) -> float:
    """Two S×S tensors (scores + weights) in FP32."""
    return 2 * batch * s * s * BYTES_PER_ELEM / (1024**3)


def _first_oom(results: Iterable[BenchmarkResult]) -> BenchmarkResult | None:
    for r in results:
        if r.oom:
            return r
    return None


def _ok_results(results: Iterable[BenchmarkResult]) -> list[BenchmarkResult]:
    return [r for r in results if not r.oom]


def render_report(results: list[BenchmarkResult]) -> str:
    ok = _ok_results(results)
    oom_first = _first_oom(results)
    max_ok = max(ok, key=lambda r: r.memory_before_backward_gib or 0) if ok else None

    lines: list[str] = [
        "# Benchmarking Scaled Dot-Product Attention",
        "",
        "**硬件：** NVIDIA A800-SXM4-80GB（80 GiB HBM）。",
        "",
        "**设定：** 孤立调用 `cs336_basics.model.scaled_dot_product_attention`；"
        f"batch={B}；单头（Q/K/V 形状 `(B, S, d)`）；FP32；无 causal mask。",
        "",
        f"**代码：** `/root/.dev/ml-sys/cs336/assignment2-systems/cs336_systems/attention_operator/` · "
        f"**数据：** `{ARTIFACTS_PATH}`。",
        "",
        "---",
        "",
        "## 1. 这个实验在量什么",
        "",
        "scaled dot-product attention（缩放点积注意力，SDPA）做三件事：",
        "",
        "1. 用 Q、K 算注意力分数矩阵，形状 `(B, S, S)`；",
        "2. 对最后一维做 softmax，得到注意力权重，形状仍是 `(B, S, S)`；",
        "3. 用权重对 V 加权求和，输出 `(B, S, d)`。",
        "",
        "朴素实现会把两张 `S×S` 矩阵留在显存里供反向使用。"
        "序列长度 S 进入分母时，**算力与显存都按 S² 增长**。"
        "本实验在 A800 上扫 `(d, S)` 网格，量三件事：",
        "",
        "| 指标 | 读法 |",
        "|------|------|",
        "| 前向均值时间 | 100 轮 forward，每轮后 `cuda.synchronize` |",
        "| backward 前 `memory_allocated` | 首轮 forward 结束、backward 开始前 |",
        "| 反向均值时间 | 100 轮 backward，每轮后 `cuda.synchronize` |",
        "",
        "每一轮是 **forward → backward → zero_grad**，对应训练步里 attention 子步的完整代价。",
        "",
        "---",
        "",
        "## 2. 显存账本（第一性原理）",
        "",
        "反向需要的前向保存项（主导项）：",
        "",
        "| 张量 | 形状 | 字节 |",
        "|------|------|------|",
        f"| attention scores | `(B, S, S)` | `B·S²·{BYTES_PER_ELEM}` |",
        f"| attention weights | `(B, S, S)` | `B·S²·{BYTES_PER_ELEM}` |",
        f"| Q, K, V | 各 `(B, S, d)` | `3·B·S·d·{BYTES_PER_ELEM}` |",
        "",
        f"两张 `S×S` 合计 **2·B·S²·{BYTES_PER_ELEM}** 字节。"
        f"在本实验 B={B} 时：",
        "",
        "```",
        "saved_GiB ≈ 2 · B · S² · 4 / 1024³",
        "```",
        "",
        "S 翻倍 → 保存项约 ×4；d 只线性进入 Q/K/V，**远小于 S² 项**。"
        "因此图里四条 d 曲线的显存几乎重合。",
        "",
        "---",
        "",
        "## 3. 时间结果",
        "",
        _img("sdpa_forward_time_vs_S.png"),
        "",
        "**前向：** QKᵀ 与后续的权重×V 都涉及 `(B, S, S)` 规模的张量运算。"
        "S 在对数横轴上每翻一倍，耗时大致向上拱——符合 S² 算力尺度。",
        "",
        _img("sdpa_backward_time_vs_S.png"),
        "",
        "**反向：** 反向要读回两张 `S×S` 保存张量并传播梯度。"
        "同样随 S 平方级变长；同一条 d 曲线上，反向普遍比前向更慢。",
        "",
        "---",
        "",
        "## 4. 显存结果",
        "",
        _img("sdpa_memory_before_backward_vs_S.png"),
        "",
        "纵轴是 backward 前的 `memory_allocated`。"
        "曲线随 S 上升，形状接近 S²；"
        "四条 d 线叠在一起，因为主导项与 d 无关。",
        "",
        "灰色虚线是 A800 的 80 GiB 上限。",
        "",
        "---",
        "",
        "## 5. 网格总览与 OOM",
        "",
        _img("sdpa_grid_summary.png", width=700),
        "",
    ]

    lines.extend(_results_table(results))
    lines.append("")

    if oom_first is not None:
        lines.extend(_section_oom(oom_first, results))
    else:
        lines.extend(_section_no_oom(max_ok, results))

    lines.extend(
        [
            "---",
            "",
            "## 7. 如何消除 S² 瓶颈",
            "",
            "根因是 **物化完整的 `(B, S, S)` 注意力矩阵**。"
            "FlashAttention 在 GPU SRAM 上分块计算 softmax，"
            "避免把整张 `S×S` 写入 HBM，"
            "同时减少 HBM 读写，前向与反向都会加速。",
            "",
            "本题测的是朴素实现的基线；"
            "下一题用融合实现替换后，"
            "同一张 `(d, S)` 网格上的时间与显存曲线应整体下移。",
            "",
        ]
    )

    return "\n".join(lines)


def _results_table(results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "### 实测表",
        "",
        "| d | S | forward (ms) | backward (ms) | mem before bwd (GiB) | OOM |",
        "|--:|--:|--:|--:|--:|:---:|",
    ]
    for r in sorted(results, key=lambda x: (x.d_model, x.seq_len)):
        if r.oom:
            lines.append(f"| {r.d_model} | {r.seq_len} | — | — | — | yes |")
        else:
            lines.append(
                f"| {r.d_model} | {r.seq_len} | {r.forward_ms:.2f} | {r.backward_ms:.2f} | "
                f"{r.memory_before_backward_gib:.3f} | no |"
            )
    return lines


def _section_oom(oom: BenchmarkResult, results: list[BenchmarkResult]) -> list[str]:
    theory = _theoretical_saved_gib(oom.seq_len)
    prior = [r for r in results if not r.oom and r.d_model == oom.d_model and r.seq_len < oom.seq_len]
    prior_peak = prior[-1].memory_before_backward_gib if prior else None
    prior_s = prior[-1].seq_len if prior else None

    lines = [
        f"### 6.1 首次 OOM：d={oom.d_model}, S={oom.seq_len}",
        "",
        f"该格在 forward 或 backward 阶段触发 CUDA OOM。",
    ]
    if prior_s is not None and prior_peak is not None:
        lines.append(
            f"同一 d={oom.d_model} 下，最后一个成功格是 S={prior_s}（{prior_peak:.3f} GiB）。"
        )
    lines.extend(
        [
            "",
            "**显存账本（最小 OOM 配置）：**",
            "",
            f"- 两张 `S×S` 保存张量理论值：2·{B}·{oom.seq_len}²·4 / 1024³ ≈ **{theory:.2f} GiB**",
            f"- Q/K/V 三项：3·{B}·{oom.seq_len}·{oom.d_model}·4 / 1024³ ≈ "
            f"**{3 * B * oom.seq_len * oom.d_model * BYTES_PER_ELEM / (1024**3):.3f} GiB**",
            "- backward 中的临时梯度缓冲与算子工作区叠加上限，总占用超过 80 GiB",
            "",
            "S 继续增大时，保存项按 S² 增长，"
            "OOM 边界因此出现在网格后段的大 S 区域。",
            "",
        ]
    )
    return lines


def _section_no_oom(max_ok: BenchmarkResult | None, results: list[BenchmarkResult]) -> list[str]:
    lines = [
        "### 6.1 本网格内全部成功",
        "",
        "在题面扩展网格上，每个 `(d, S)` 配置均完成 100 轮 forward+backward。",
        "",
    ]
    if max_ok:
        theory = _theoretical_saved_gib(max_ok.seq_len)
        lines.extend(
            [
                f"显存最高的一格：d={max_ok.d_model}, S={max_ok.seq_len}，"
                f"backward 前 **{max_ok.memory_before_backward_gib:.3f} GiB**。"
                f"两张 `S×S` 理论保存 **{theory:.2f} GiB**，"
                "与实测同量级。",
                "",
                "在 B=8、FP32、孤立算子设定下，"
                "S 需继续增大才会在 80 GiB 卡上触顶；"
                "机制仍是 S² 保存项最终占满 HBM。",
                "",
            ]
        )
    return lines
