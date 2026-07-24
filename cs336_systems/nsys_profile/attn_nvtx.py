"""NVTX wrappers around attention SDPA matmul vs softmax (no basics edits)."""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.cuda.nvtx as nvtx
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor

import cs336_basics.model as model_mod
from cs336_basics.nn_utils import softmax

NVTX_ATTN_MATMUL = "attn_matmul"
NVTX_ATTN_SOFTMAX = "attn_softmax"

_original_sdpa: Callable | None = None


def _sdpa_with_nvtx(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Same math as basics SDPA; NVTX splits attention matmuls vs softmax."""
    d_k = K.shape[-1]
    with nvtx.range(NVTX_ATTN_MATMUL):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range(NVTX_ATTN_SOFTMAX):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx.range(NVTX_ATTN_MATMUL):
        return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


def install_attn_nvtx() -> None:
    """Monkey-patch basics SDPA with NVTX-annotated version."""
    global _original_sdpa
    if _original_sdpa is not None:
        return
    _original_sdpa = model_mod.scaled_dot_product_attention
    model_mod.scaled_dot_product_attention = _sdpa_with_nvtx


def uninstall_attn_nvtx() -> None:
    global _original_sdpa
    if _original_sdpa is None:
        return
    model_mod.scaled_dot_product_attention = _original_sdpa
    _original_sdpa = None
