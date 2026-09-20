#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 30 Threads on lab1 (48 threads total)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=30
export MKL_NUM_THREADS=30
export OPENBLAS_NUM_THREADS=30
export VECLIB_MAXIMUM_THREADS=30
export NUMEXPR_NUM_THREADS=30

echo "=========================================================="
echo "Starting CPU-only benchmark (20 epochs): 30 Threads"
echo "Affinity mask: taskset -c 0-29"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-29 python train.py --config configs/cpu/cpu_30cores_20e.yaml

