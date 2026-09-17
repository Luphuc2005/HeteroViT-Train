#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# GPU benchmark (20 epochs, batch_size=256) constrained to 8 physical CPU cores
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=2

echo "=========================================================="
echo "Starting GPU benchmark: 1 GPU | 8 CPU Cores | Batch Size: 256 | Epochs: 20"
echo "Affinity mask: taskset -c 0-7"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "OMP_NUM_THREADS: '$OMP_NUM_THREADS'"
echo "TF_NUM_INTRAOP_THREADS: '$TF_NUM_INTRAOP_THREADS' | TF_NUM_INTEROP_THREADS: '$TF_NUM_INTEROP_THREADS'"
echo "=========================================================="

taskset -c 0-7 python train.py --config configs/gpu/gpu_cpu8_b256_20e.yaml
