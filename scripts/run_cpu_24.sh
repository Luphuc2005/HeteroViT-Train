#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU baseline: 24 cores on single GPU node (disabling GPU)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

echo "=========================================================="
echo "Starting CPU-only benchmark: 24 Cores"
echo "Affinity mask: taskset -c 0-23"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-23 python train.py --config configs/cpu/cpu_24cores.yaml
