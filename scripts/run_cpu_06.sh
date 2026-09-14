#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU baseline: 6 cores on single GPU node (disabling GPU)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

echo "=========================================================="
echo "Starting CPU-only benchmark: 6 Cores"
echo "Affinity mask: taskset -c 0-5"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-5 python train.py --config configs/cpu/cpu_06cores.yaml
