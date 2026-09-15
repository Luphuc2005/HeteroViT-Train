#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 12 Cores / 12 Threads
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export OPENBLAS_NUM_THREADS=12
export VECLIB_MAXIMUM_THREADS=12
export NUMEXPR_NUM_THREADS=12

echo "=========================================================="
echo "Starting CPU-only benchmark (20 epochs): 12 Cores / 12 Threads"
echo "Affinity mask: taskset -c 0-11"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-11 python train.py --config configs/cpu/cpu_12cores_20e.yaml
