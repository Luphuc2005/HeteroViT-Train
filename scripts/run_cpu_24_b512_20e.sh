#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 24 Cores / 24 Threads | Batch Size: 512
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24
export VECLIB_MAXIMUM_THREADS=24
export NUMEXPR_NUM_THREADS=24

echo "=========================================================="
echo "Starting CPU-only benchmark: 24 Cores | Batch Size: 512"
echo "Affinity mask: taskset -c 0-23"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-23 python train.py --config configs/cpu/cpu_24cores_b512_20e.yaml
