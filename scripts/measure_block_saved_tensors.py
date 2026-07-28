#!/usr/bin/env python3
"""Measure saved-for-backward tensors in one TransformerBlock (handout §3.2).

Reproduces the ``saved_tensors_hooks`` tally that reports ~3651 MiB for a
compiled xl-style block at B=4, S=2048.

Usage:

  uv run --no-sync python /root/.dev/ml-sys/cs336/assignment2-systems/scripts/measure_block_saved_tensors.py
  uv run --no-sync python /root/.dev/ml-sys/cs336/assignment2-systems/scripts/measure_block_saved_tensors.py --verbose
  uv run --no-sync python /root/.dev/ml-sys/cs336/assignment2-systems/scripts/measure_block_saved_tensors.py --no-compile
  uv run --no-sync python /root/.dev/ml-sys/cs336/assignment2-systems/scripts/measure_block_saved_tensors.py --context-length 512
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from cs336_basics.model import RotaryEmbedding, TransformerBlock


def _mib(nbytes: int) -> float:
    return nbytes / (1024**2)


def _gib(nbytes: int) -> float:
    return nbytes / (1024**3)


@dataclass
class SavedTensorRecord:
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int
    grad_fn: str | None
    bucket: str


@dataclass
class SavedTensorReport:
    records: list[SavedTensorRecord] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(r.nbytes for r in self.records)

    def by_bucket(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for r in self.records:
            out[r.bucket] += r.nbytes
        return dict(out)

    def by_shape(self) -> dict[tuple[int, ...], int]:
        out: dict[tuple[int, ...], int] = defaultdict(int)
        for r in self.records:
            out[r.shape] += r.nbytes
        return dict(out)


def classify_tensor(
    t: torch.Tensor,
    *,
    batch: int,
    context: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
) -> str:
    if isinstance(t, nn.Parameter):
        return "parameter"
    nbytes = t.numel() * t.element_size()
    shape = tuple(t.shape)
    mib = _mib(nbytes)
    d_k = d_model // num_heads

    attn_ss = batch * num_heads * context * context * 4
    ffn_act = batch * context * d_ff * 4
    hid = batch * context * d_model * 4
    qkv = batch * num_heads * context * d_k * 4
    ffn_w = d_model * d_ff * 4
    attn_w = d_model * d_model * 4

    if len(shape) >= 2 and sorted(shape[-2:]) == sorted([d_model, d_ff]):
        return "ffn_weight"
    if len(shape) == 2 and shape == (d_model, d_model):
        return "attn_weight"
    if nbytes == attn_ss or (
        len(shape) >= 2 and shape[-1] == context and shape[-2] == context and mib >= 50
    ):
        return "attn_SxS"
    if nbytes == ffn_act or (len(shape) >= 1 and shape[-1] == d_ff and abs(mib - _mib(ffn_act)) < 2):
        return "ffn_inner_(B,S,d_ff)"
    if nbytes == qkv or (len(shape) >= 1 and shape[-1] == d_k and abs(mib - _mib(qkv)) < 2):
        return "attn_qkv_heads"
    if nbytes == hid or (len(shape) >= 1 and shape[-1] == d_model and abs(mib - _mib(hid)) < 2):
        return "hidden_(B,S,d)"
    if nbytes == ffn_w:
        return "ffn_weight"
    if nbytes == attn_w:
        return "attn_weight"
    if mib < 1.0:
        return "small_(norm_stats_etc)"
    return f"other{shape}"


def measure_block(
    *,
    batch: int,
    context: int,
    d_model: int,
    d_ff: int,
    num_heads: int,
    compile_block: bool,
    device: torch.device,
) -> SavedTensorReport:
    positional_encoder = RotaryEmbedding(
        context_length=context,
        dim=d_model // num_heads,
    )
    block = TransformerBlock(
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        positional_encoder=positional_encoder,
    ).to(device)
    if compile_block:
        block = torch.compile(block, fullgraph=True)

    x = torch.randn(batch, context, d_model, device=device, requires_grad=True)
    report = SavedTensorReport()

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, nn.Parameter):
            return t
        nbytes = t.numel() * t.element_size()
        grad_fn = None if t.grad_fn is None else type(t.grad_fn).__name__
        bucket = classify_tensor(
            t,
            batch=batch,
            context=context,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
        )
        report.records.append(
            SavedTensorRecord(
                shape=tuple(t.shape),
                dtype=t.dtype,
                nbytes=nbytes,
                grad_fn=grad_fn,
                bucket=bucket,
            )
        )
        return t

    def unpack_hook(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = block(x)
    # Touch output so the graph is real; handout stops at forward only.
    _ = y.sum()

    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="Tally saved tensors for one TransformerBlock (handout §3.2)",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=2560)
    p.add_argument("--d-ff", type=int, default=10240)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Skip torch.compile (compare against fused block)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print every saved tensor as it is registered",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA required (handout uses GPU + torch.compile).")

    device = torch.device(args.device)
    compile_block = not args.no_compile

    print("── Handout §3.2: saved tensors in one TransformerBlock ──")
    print(f"  batch={args.batch_size}  context={args.context_length}  d_model={args.d_model}")
    print(f"  d_ff={args.d_ff}  num_heads={args.num_heads}  compile={compile_block}")
    print(f"  device={device}")
    print()

    # Reference single-tensor sizes for this shape
    B, S, d, d_ff, H = (
        args.batch_size,
        args.context_length,
        args.d_model,
        args.d_ff,
        args.num_heads,
    )
    print("── Reference: one tensor of each shape (FP32) ──")
    print(f"  hidden (B,S,d)     = {_mib(B * S * d * 4):.2f} MiB")
    print(f"  attn S×S (B,H,S,S) = {_mib(B * H * S * S * 4):.2f} MiB")
    print(f"  ffn inner (B,S,d_ff)= {_mib(B * S * d_ff * 4):.2f} MiB")
    print(f"  attn weight (d,d)  = {_mib(d * d * 4):.2f} MiB")
    print(f"  ffn weight (d,d_ff)= {_mib(d * d_ff * 4):.2f} MiB")
    print()

    report = measure_block(
        batch=B,
        context=S,
        d_model=d,
        d_ff=d_ff,
        num_heads=H,
        compile_block=compile_block,
        device=device,
    )

    total = report.total_bytes
    by_bucket = report.by_bucket()
    bucket_counts: dict[str, int] = defaultdict(int)
    for r in report.records:
        bucket_counts[r.bucket] += 1

    if args.verbose:
        print("── Every saved tensor (forward registration order) ──")
        for i, r in enumerate(report.records, 1):
            print(
                f"  {i:3d}. {_mib(r.nbytes):8.2f} MiB  {str(r.shape):<28} "
                f"{r.bucket:<22} grad_fn={r.grad_fn}"
            )
        print()

    rows = sorted(by_bucket.items(), key=lambda kv: kv[1], reverse=True)
    print("── By category (where the total comes from) ──")
    print(f"{'category':<28} {'MiB':>12} {'%':>8} {'#tensors':>10}")
    print("-" * 62)
    for name, nbytes in rows:
        print(
            f"{name:<28} {_mib(nbytes):>12.2f} {100 * nbytes / total:>7.1f}% "
            f"{bucket_counts[name]:>10d}"
        )
    print("-" * 62)
    print(
        f"{'TOTAL':<28} {_mib(total):>12.2f} {100.0:>7.1f}% "
        f"{len(report.records):>10d}"
    )
    print()
    print(
        f"Total size of saved tensors in single TransformerBlock: "
        f"{_mib(total):.2f} MiB ({_gib(total):.3f} GiB)"
    )
    if compile_block and S == 2048:
        print(
            "(handout §3.2 reports ~3651 MiB; exact MiB varies slightly with "
            "PyTorch / inductor version, same ~3.5–4.0 GiB ballpark)"
        )

    print("\n── Plain-language stack-up (why ~3.5 GiB) ──")
    attn = by_bucket.get("attn_SxS", 0)
    ffn = by_bucket.get("ffn_inner_(B,S,d_ff)", 0)
    qkv = by_bucket.get("attn_qkv_heads", 0)
    weights = by_bucket.get("ffn_weight", 0) + by_bucket.get("attn_weight", 0)
    print(f"  1. Attention S×S maps     {_mib(attn):>10.2f} MiB  "
          f"({bucket_counts.get('attn_SxS', 0)} tensors; each B·H·S·S at S={S} is "
          f"{_mib(B * H * S * S * 4):.0f} MiB)")
    print(f"  2. FFN wide (B,S,d_ff)     {_mib(ffn):>10.2f} MiB  "
          f"({bucket_counts.get('ffn_inner_(B,S,d_ff)', 0)} tensors; each is "
          f"{_mib(B * S * d_ff * 4):.0f} MiB)")
    print(f"  3. Q/K/V head activations  {_mib(qkv):>10.2f} MiB")
    print(f"  4. Weight refs in hooks    {_mib(weights):>10.2f} MiB  "
          "(parameters already resident; hooks count shape bytes)")

    by_shape = report.by_shape()
    top_shapes = sorted(by_shape.items(), key=lambda kv: kv[1], reverse=True)[:12]
    print("\n── Top shapes by total MiB ──")
    for shape, nbytes in top_shapes:
        n = sum(1 for r in report.records if r.shape == shape)
        print(f"  {str(shape):<30} {_mib(nbytes):>10.2f} MiB  (×{n})")


if __name__ == "__main__":
    main()
