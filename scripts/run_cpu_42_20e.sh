#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CPU benchmark (20 epochs): 42 Threads on lab1 (48 threads total)
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=42
export MKL_NUM_THREADS=42
export OPENBLAS_NUM_THREADS=42
export VECLIB_MAXIMUM_THREADS=42
export NUMEXPR_NUM_THREADS=42

echo "=========================================================="
echo "Starting CPU-only benchmark (20 epochs): 42 Threads"
echo "Affinity mask: taskset -c 0-41"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

taskset -c 0-41 python train.py --config configs/cpu/cpu_42cores_20e.yaml

