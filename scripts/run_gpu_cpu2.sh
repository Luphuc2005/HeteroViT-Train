#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# GPU benchmark (20 epochs, batch_size=256) constrained to 2 physical CPU cores
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2

echo "=========================================================="
echo "Starting GPU benchmark: 1 GPU | 2 CPU Cores | Batch Size: 256 | Epochs: 20"
echo "Affinity mask: taskset -c 0-1"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "OMP_NUM_THREADS: '$OMP_NUM_THREADS'"
echo "TF_NUM_INTRAOP_THREADS: '$TF_NUM_INTRAOP_THREADS' | TF_NUM_INTEROP_THREADS: '$TF_NUM_INTEROP_THREADS'"
echo "=========================================================="

taskset -c 0-1 python train.py --config configs/gpu/gpu_cpu2_b256_20e.yaml
