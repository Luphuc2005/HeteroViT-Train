#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 36 Threads on lab1 (48 threads total)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=36
export MKL_NUM_THREADS=36
export OPENBLAS_NUM_THREADS=36
export VECLIB_MAXIMUM_THREADS=36
export NUMEXPR_NUM_THREADS=36

echo "=========================================================="
echo "Starting CPU-only benchmark (20 epochs): 36 Threads"
echo "Affinity mask: taskset -c 0-35"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-35 python train.py --config configs/cpu/cpu_36cores_20e.yaml

