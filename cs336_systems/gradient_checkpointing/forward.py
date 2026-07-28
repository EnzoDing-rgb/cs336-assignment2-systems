"""Checkpointed forward over Transformer layers only (embed / head unchanged)."""

from __future__ import annotations

import torch
from cs336_basics.model import BasicsTransformerLM
from torch.utils.checkpoint import checkpoint


def _run_layer_range(
    x: torch.Tensor,
    layers: torch.nn.ModuleList,
    start: int,
    end: int,
) -> torch.Tensor:
    for i in range(start, end):
        x = layers[i](x)
    return x


def forward_lm_with_checkpoint(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    segment_size: int | None,
) -> torch.Tensor:
    """Forward through the LM; checkpoint applies only to ``model.layers``.

    segment_size:
        None — plain sequential forward (baseline).
        k     — each contiguous k layers wrapped in one ``checkpoint`` call.
    """
    h = model.token_embeddings(token_ids)
    layers = model.layers
    n_layers = len(layers)

    if segment_size is None:
        for layer in layers:
            h = layer(h)
    else:
        if segment_size < 1:
            raise ValueError(f"segment_size must be >= 1, got {segment_size}")
        i = 0
        while i < n_layers:
            end = min(i + segment_size, n_layers)
            start, stop = i, end

            def run_segment(inp: torch.Tensor, s: int = start, e: int = stop) -> torch.Tensor:
                return _run_layer_range(inp, layers, s, e)

            h = checkpoint(run_segment, h, use_reentrant=False)
            i = end

    h = model.ln_final(h)
    return model.lm_head(h)
