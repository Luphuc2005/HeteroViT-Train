#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 24 Cores / 48 Threads (Hyper-Threading)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export VECLIB_MAXIMUM_THREADS=48
export NUMEXPR_NUM_THREADS=48

echo "=========================================================="
echo "Starting CPU-only benchmark (20 epochs): 24 Cores / 48 Threads (Hyper-Threading)"
echo "Affinity mask: taskset -c 0-47"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-47 python train.py --config configs/cpu/cpu_48t_20e.yaml
