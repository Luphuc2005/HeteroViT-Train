#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Ensure libdevice.10.bc for XLA if available
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

echo "============================================================================"
echo ">>> Joint Training Option A (In-Place Zero-Allocation RAM Merge, Iso-Batch 256)"
echo "Workload: GPU 232 (116/GPU) + CPU 24 (18 Cores) = 256 (20 Epochs)"
echo "Config: configs/joint/joint_2gpu_b232_cpu18_b24_optA_20e.yaml"
echo "============================================================================"

export PYTHONUNBUFFERED=1
python -u train.py --config configs/joint/joint_2gpu_b232_cpu18_b24_optA_20e.yaml
