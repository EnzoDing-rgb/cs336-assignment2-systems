#!/bin/bash
# =============================================================================
# Naive DDP Benchmark — Full Sweep
#
# Each configuration runs as a completely separate `uv run` invocation.
# This guarantees the OS reclaims all GPU memory between runs — no CUDA IPC
# leaks, no cross-run contamination.
#
# Output:
#   artifacts/naive_ddp_benchmark.csv
#   artifacts/naive_ddp_per_param_latency.csv
# =============================================================================

set -euo pipefail

SCRIPT="cs336_systems.distributed.benchmarking.benchmark_naive_ddp"
OUTPUT="artifacts/naive_ddp_benchmark.csv"
PER_PARAM="artifacts/naive_ddp_per_param_latency.csv"

# Clean start
rm -f "$OUTPUT" "$PER_PARAM"

echo "============================================"
echo "Phase 1: Single-GPU baseline (batch=4, 8)"
echo "============================================"

for bs in 4 8; do
    echo ""
    echo ">>> Single-GPU batch=$bs"
    uv run python -m "$SCRIPT" \
        --batch-sizes "$bs" \
        --skip-ddp \
        --no-per-param \
        --output "$OUTPUT"
done

echo ""
echo "============================================"
echo "Phase 2: 2-GPU NaiveDDP (batch=4, 8, 16)"
echo "============================================"

for bs in 4 8 16; do
    echo ""
    echo ">>> NaiveDDP batch=$bs"
    if [ "$bs" = "4" ]; then
        # Record per-parameter latency on first DDP run only
        uv run python -m "$SCRIPT" \
            --batch-sizes "$bs" \
            --skip-single \
            --output "$OUTPUT"
    else
        uv run python -m "$SCRIPT" \
            --batch-sizes "$bs" \
            --skip-single \
            --no-per-param \
            --output "$OUTPUT"
    fi
done

echo ""
echo "============================================"
echo "Done!"
echo "  Data:    $OUTPUT"
echo "  Per-param: $PER_PARAM"
echo "============================================"
