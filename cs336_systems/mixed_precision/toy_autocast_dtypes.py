"""ToyModel + autocast: print dtypes for Assignment 2 benchmarking_mixed_precision (a)(b).

Runs three regimes on CUDA:
  1) no autocast (full FP32)
  2) autocast FP16
  3) autocast BF16

Prints full descriptive labels (not abbreviations) for parameters, layer outputs,
loss, and gradients.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "mixed_precision"
OUT_JSON = ARTIFACT_DIR / "toy_autocast_dtypes.json"


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def _dtype_name(t: torch.Tensor | None) -> str:
    if t is None:
        return "None"
    return str(t.dtype).replace("torch.", "")


def _run_one(
    *,
    label: str,
    autocast_dtype: torch.dtype | None,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(0)
    model = ToyModel(in_features=16, out_features=4).to(device)
    # Parameters stay FP32 unless we explicitly cast the module.
    assert all(p.dtype == torch.float32 for p in model.parameters())

    x = torch.randn(8, 16, device=device, dtype=torch.float32)
    ctx = (
        nullcontext()
        if autocast_dtype is None
        else torch.autocast(device_type="cuda", dtype=autocast_dtype)
    )

    # Peek parameter dtype *inside* the autocast region (should still be FP32).
    with ctx:
        param_inside = _dtype_name(model.fc1.weight)

    with ctx:
        h1 = model.fc1(x)
        h1_act = model.relu(h1)
        h_ln = model.ln(h1_act)
        out = model.fc2(h_ln)
        # Fake scalar loss for dtype of loss / grads (assignment does not need real labels).
        loss = out.mean()

    loss.backward()

    row = {
        "regime": label,
        "autocast_dtype": None if autocast_dtype is None else _dtype_name(torch.tensor(0, dtype=autocast_dtype)),
        "model_parameters_outside_autocast": _dtype_name(model.fc1.weight),
        "model_parameters_inside_autocast_context": param_inside,
        "first_feedforward_ToyModel_fc1_output": _dtype_name(h1),
        "layer_norm_ToyModel_ln_output": _dtype_name(h_ln),
        "second_feedforward_ToyModel_fc2_output": _dtype_name(out),
        "loss": _dtype_name(loss),
        "gradient_of_first_feedforward_weight": _dtype_name(model.fc1.weight.grad),
        "gradient_of_layer_norm_weight": _dtype_name(model.ln.weight.grad),
    }
    return row


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for autocast dtype probe.")
    device = torch.device("cuda")
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0)}")
    print(f"bf16_supported={torch.cuda.is_bf16_supported()}")

    rows = [
        _run_one(label="no_autocast_fp32", autocast_dtype=None, device=device),
        _run_one(label="autocast_fp16", autocast_dtype=torch.float16, device=device),
        _run_one(label="autocast_bf16", autocast_dtype=torch.bfloat16, device=device),
    ]

    # Human-readable table
    keys = [
        "regime",
        "model_parameters_inside_autocast_context",
        "first_feedforward_ToyModel_fc1_output",
        "layer_norm_ToyModel_ln_output",
        "loss",
        "gradient_of_first_feedforward_weight",
    ]
    print("\n=== dtype probe (selected columns) ===")
    header = " | ".join(f"{k}" for k in keys)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
