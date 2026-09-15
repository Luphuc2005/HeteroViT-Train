#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 18 Cores / 18 Threads
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
export OPENBLAS_NUM_THREADS=18
export VECLIB_MAXIMUM_THREADS=18
export NUMEXPR_NUM_THREADS=18

echo "=========================================================="
echo "Starting CPU-only benchmark (20 epochs): 18 Cores / 18 Threads"
echo "Affinity mask: taskset -c 0-17"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-17 python train.py --config configs/cpu/cpu_18cores_20e.yaml
