#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Locate libdevice.10.bc for XLA if available
if [ ! -f "libdevice.10.bc" ]; then
    for cand in \
        /usr/local/cuda/nvvm/libdevice/libdevice.10.bc \
        /usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc \
        /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc \
        "$CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc" \
        "$VIRTUAL_ENV/lib/python*/site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc"; do
        if [ -f "$cand" ]; then
            ln -sf "$cand" ./libdevice.10.bc 2>/dev/null || true
            break
        fi
    done
fi

# GPU benchmark (20 epochs, batch_size=256) constrained to 4 physical CPU cores
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=2

echo "=========================================================="
echo "Starting GPU benchmark: 1 GPU | 4 CPU Cores | Batch Size: 256 | Epochs: 20"
echo "Affinity mask: taskset -c 0-3"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "OMP_NUM_THREADS: '$OMP_NUM_THREADS'"
echo "TF_NUM_INTRAOP_THREADS: '$TF_NUM_INTRAOP_THREADS' | TF_NUM_INTEROP_THREADS: '$TF_NUM_INTEROP_THREADS'"
echo "=========================================================="

taskset -c 0-3 python train.py --config configs/gpu/gpu_cpu4_b256_20e.yaml
