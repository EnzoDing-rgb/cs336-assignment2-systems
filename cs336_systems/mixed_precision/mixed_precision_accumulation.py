"""Assignment 2: mixed_precision_accumulation.

Run the four accumulation snippets from the handout and print results.
You inspect the CLI output yourself; the writeup comment comes after.
"""

from __future__ import annotations

import torch


def run_cases() -> None:
    # (1) FP32 accumulator += FP32 addend
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print("case1_fp32_acc_fp32_add:", s)

    # (2) FP16 accumulator += FP16 addend
    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("case2_fp16_acc_fp16_add:", s)

    # (3) FP32 accumulator += FP16 addend (implicit promotion on +=)
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print("case3_fp32_acc_fp16_add:", s)

    # (4) FP32 accumulator += explicitly cast FP16 -> FP32
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print("case4_fp32_acc_cast_fp16_to_fp32:", s)


def main() -> None:
    print(f"torch={torch.__version__}")
    print("expected_exact_sum_if_perfect=10.0  # 1000 * 0.01")
    run_cases()


if __name__ == "__main__":
    main()
