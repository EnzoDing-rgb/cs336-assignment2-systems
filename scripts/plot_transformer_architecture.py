#!/usr/bin/env python3
"""Draw BasicsTransformerLM architecture diagram for memory-profiling report.

Usage:
  uv run --no-sync python scripts/plot_transformer_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "reports" / "figures" / "memory_architecture_basics_transformer_lm.png"

_CJK_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "WenQuanYi Zen Hei",
]


def _setup_font() -> str:
    for cand in _CJK_CANDIDATES:
        try:
            if cand.startswith("/") and Path(cand).exists():
                font_manager.fontManager.addfont(cand)
                name = font_manager.FontProperties(fname=cand).get_name()
                plt.rcParams["font.family"] = name
                plt.rcParams["axes.unicode_minus"] = False
                return name
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def _box(ax, cx, y_top, w, h, text, *, fc, ec="#2B2B2B", lw=1.15, fs=8.0, weight="normal"):
    x = cx - w / 2
    y = y_top - h
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.035",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        cx,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight=weight,
        color="#1A1A1A",
        linespacing=1.2,
    )
    return y  # new y_top for next (bottom of this box)


def _arrow(ax, cx, y_from, y_to):
    ax.annotate(
        "",
        xy=(cx, y_to),
        xytext=(cx, y_from),
        arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.25, shrinkA=0, shrinkB=0),
    )


def main() -> None:
    font_name = _setup_font()
    print(f"Using font: {font_name}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Coordinate system: y decreases downward in drawing order, but matplotlib y grows up.
    # We allocate a tall canvas and place from top.
    fig_w, fig_h = 11.0, 18.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 18)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    cx = 5.5
    box_w = 7.0
    gap = 0.14
    y = 17.55

    ax.text(
        cx,
        y,
        "BasicsTransformerLM · xl 架构（理论示意）",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    y -= 0.38
    ax.text(
        cx,
        y,
        r"$B{=}4,\ S{=}512,\ d{=}2560,\ d_{ff}{=}10240,\ H{=}32,\ L{=}32,\ V{=}10000$",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )
    y -= 0.55

    # Token IDs
    h = 0.70
    _arrow(ax, cx, y + 0.05, y) if False else None
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        "Token IDs\n形状 $(B,S)$ · 例 $(4,512)$",
        fc="#EEF3F8",
        weight="bold",
        fs=8.5,
    )
    y -= gap
    _arrow(ax, cx, y + gap, y)

    # Embedding
    h = 0.80
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        "Token Embedding · Embedding$(V,d)$\n"
        r"$(B,S)\rightarrow(B,S,d)$ · 例 $(4,512,2560)$",
        fc="#E8F0E8",
        weight="bold",
        fs=8.3,
    )
    y_after_emb = y
    y -= gap + 0.08
    _arrow(ax, cx, y_after_emb, y)

    # ---- Measure block content first by placing, then draw frames behind via zorder ----
    # We'll draw frames with zorder=0 and boxes with zorder=2.

    block_inner_top = y
    block_pad_top = 0.42
    y -= block_pad_top
    ax.text(
        1.15,
        block_inner_top - 0.18,
        r"TransformerBlock × $L{=}32$（每层重复）",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="#1F4E79",
        zorder=3,
    )

    # residual in
    h = 0.58
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        r"残差流输入 $x$ · $(B,S,d)$   ← 概念 2",
        fc="#FFF7E6",
        weight="bold",
        fs=8.3,
    )
    y -= gap
    _arrow(ax, cx, y + gap, y)

    # Attention section header space
    attn_label_y = y
    y -= 0.32
    ax.text(
        1.35,
        attn_label_y - 0.12,
        r"Attention 子层  $=\,x + \mathrm{Attn}(\mathrm{LN}(x))$",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#5A3E1B",
        zorder=3,
    )
    attn_content_top = y

    attn_steps = [
        (0.58, "RMSNorm ln1\n$(B,S,d)\\rightarrow(B,S,d)$", "#F3EDE3"),
        (
            0.78,
            "线性投影 q_proj / k_proj / v_proj · 各 $(d\\rightarrow d)$\n"
            r"reshape → $Q,K,V\in(B,H,S,d_k)$ · $d_k{=}d/H{=}80$",
            "#F3EDE3",
        ),
        (0.58, "RoPE · 旋转 $Q$ 与 $K$\n形状不变 $(B,H,S,d_k)$", "#F3EDE3"),
        (
            1.00,
            "scaled_dot_product_attention\n"
            r"$\mathrm{scores}=QK^{\top}/\sqrt{d_k}\in(B,H,S,S)$ · 单张 $128$ MiB"
            "\nsoftmax → 权重，再 × $V$ → $(B,H,S,d_k)$",
            "#F8E8D0",
        ),
        (0.62, "合并头 + output_proj\n$(B,S,d)\\rightarrow(B,S,d)$", "#F3EDE3"),
        (0.58, r"残差加法  $x + x_{\mathrm{attn}}$   ← 概念 1", "#FFE8C8"),
    ]
    for i, (h, text, fc) in enumerate(attn_steps):
        y = _box(ax, cx, y, box_w, h, text, fc=fc, fs=8.0)
        if i < len(attn_steps) - 1:
            y -= gap
            _arrow(ax, cx, y + gap, y)
    attn_content_bot = y

    y -= gap + 0.06
    _arrow(ax, cx, attn_content_bot, y)

    # FFN header
    ffn_label_y = y
    y -= 0.32
    ax.text(
        1.35,
        ffn_label_y - 0.12,
        r"FFN 子层  $=\,x' + \mathrm{SwiGLU}(\mathrm{LN}(x'))$",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="#1E4A37",
        zorder=3,
    )
    ffn_content_top = y

    h = 0.55
    y = _box(ax, cx, y, box_w, h, "RMSNorm ln2\n$(B,S,d)\\rightarrow(B,S,d)$", fc="#E5F3EA", fs=8.0)
    y_after_ln2 = y
    y -= 0.22
    _arrow(ax, cx, y_after_ln2, y)

    # parallel w1 / w3
    h = 0.72
    w_small = 3.2
    y_top_par = y
    # split lines
    ax.plot([cx, cx], [y_after_ln2, y_top_par + 0.02], color="#333", lw=1.15, zorder=1)
    ax.plot([cx - 1.7, cx + 1.7], [y_top_par + 0.02, y_top_par + 0.02], color="#333", lw=1.15, zorder=1)
    _box(
        ax,
        cx - 1.75,
        y_top_par,
        w_small,
        h,
        r"$w_1:(d\rightarrow d_{ff})$" + "\n" + r"$(B,S,d_{ff})$ · $80$ MiB",
        fc="#E5F3EA",
        fs=7.8,
    )
    _box(
        ax,
        cx + 1.75,
        y_top_par,
        w_small,
        h,
        r"$w_3:(d\rightarrow d_{ff})$" + "\n" + r"$(B,S,d_{ff})$ · $80$ MiB",
        fc="#E5F3EA",
        fs=7.8,
    )
    _arrow(ax, cx - 1.75, y_top_par + 0.02, y_top_par)
    _arrow(ax, cx + 1.75, y_top_par + 0.02, y_top_par)
    y = y_top_par - h - 0.12
    # merge
    ax.plot([cx - 1.75, cx - 1.75], [y_top_par - h, y + 0.05], color="#333", lw=1.1, zorder=1)
    ax.plot([cx + 1.75, cx + 1.75], [y_top_par - h, y + 0.05], color="#333", lw=1.1, zorder=1)
    ax.plot([cx - 1.75, cx + 1.75], [y + 0.05, y + 0.05], color="#333", lw=1.1, zorder=1)
    ax.plot([cx, cx], [y + 0.05, y], color="#333", lw=1.1, zorder=1)

    h = 0.70
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        r"$\mathrm{silu}(w_1(x))\odot w_3(x)$" + "\n" + r"仍 $(B,S,d_{ff})$ · $80$ MiB",
        fc="#D9EFDF",
        fs=8.0,
    )
    y -= gap
    _arrow(ax, cx, y + gap, y)

    h = 0.58
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        r"$w_2:(d_{ff}\rightarrow d)$ · $\rightarrow(B,S,d)$",
        fc="#E5F3EA",
        fs=8.0,
    )
    y -= gap
    _arrow(ax, cx, y + gap, y)

    h = 0.58
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        r"残差加法  $x' + x_{\mathrm{ffn}}$   ← 概念 1",
        fc="#CDEAD8",
        weight="bold",
        fs=8.2,
    )
    ffn_content_bot = y

    y -= gap + 0.05
    _arrow(ax, cx, ffn_content_bot, y)

    h = 0.58
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        "残差流输出 $(B,S,d)$ → 下一层 / 最终 LN",
        fc="#FFF7E6",
        weight="bold",
        fs=8.2,
    )
    block_content_bot = y
    block_bottom = block_content_bot - 0.18
    block_top = block_inner_top + 0.08

    # Draw frames behind
    ax.add_patch(
        FancyBboxPatch(
            (0.7, block_bottom),
            9.6,
            block_top - block_bottom,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6,
            edgecolor="#2F5D8A",
            facecolor="#F4F8FC",
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.05, attn_content_bot - 0.12),
            8.9,
            (attn_label_y + 0.05) - (attn_content_bot - 0.12),
            boxstyle="round,pad=0.015,rounding_size=0.05",
            linewidth=1.2,
            edgecolor="#6B4F2A",
            facecolor="#FFFBF3",
            zorder=0,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (1.05, ffn_content_bot - 0.12),
            8.9,
            (ffn_label_y + 0.05) - (ffn_content_bot - 0.12),
            boxstyle="round,pad=0.015,rounding_size=0.05",
            linewidth=1.2,
            edgecolor="#2A5A45",
            facecolor="#F3FBF6",
            zorder=0,
        )
    )

    # Repeat arrow on the right
    ax.annotate(
        "",
        xy=(10.05, block_inner_top - 0.95),
        xytext=(10.05, block_content_bot + 0.1),
        arrowprops=dict(arrowstyle="-|>", color="#2F5D8A", lw=1.4, linestyle="--"),
        zorder=3,
    )
    ax.text(
        10.25,
        (block_inner_top - 0.95 + block_content_bot + 0.1) / 2,
        "重复\n$L{-}1$\n次",
        ha="left",
        va="center",
        fontsize=8,
        color="#2F5D8A",
        fontweight="bold",
        zorder=3,
    )

    # After block
    y = block_bottom - 0.22
    _arrow(ax, cx, block_content_bot, y)

    h = 0.58
    y = _box(ax, cx, y, box_w, h, "ln_final · RMSNorm\n$(B,S,d)$", fc="#EEF3F8", fs=8.2)
    y -= gap
    _arrow(ax, cx, y + gap, y)

    h = 0.72
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        "LM Head · Linear$(d\\rightarrow V)$\n"
        r"logits $(B,S,V)$ · 例 $(4,512,10000)$ · 与 Embedding 未绑权",
        fc="#E8F0E8",
        weight="bold",
        fs=8.0,
    )
    y -= gap
    _arrow(ax, cx, y + gap, y)

    h = 0.55
    y = _box(
        ax,
        cx,
        y,
        box_w,
        h,
        "可选: cross_entropy(logits, labels) → 标量 loss",
        fc="#F5F5F5",
        fs=8.0,
    )

    ax.text(
        0.7,
        0.28,
        "颜色：黄=残差流 · 棕=Attention 子层 · 绿=FFN/SwiGLU · 蓝框=整层重复 L 次\n"
        "概念 1=残差连接 ADD；概念 2=贯穿的 $(B,S,d)$ 残差流；概念 3=子层内部为反传保存的中间张量 $R$",
        ha="left",
        va="center",
        fontsize=7.5,
        color="#333333",
    )

    # Raise all FancyBboxPatch boxes above frames: already default; re-set zorder for text boxes
    for p in ax.patches:
        if getattr(p, "get_facecolor", None) and p.get_zorder() == 1:
            p.set_zorder(2)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
