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

export PYTHONUNBUFFERED=1

echo "============================================================================"
echo ">>> [1/2] Running V5 MAXPERF Baseline with SEED 43 (20 Epochs)"
echo "Config: configs/joint/joint_2gpu_b232_cpu18_b24_maxperf_seed43.yaml"
echo "============================================================================"
python -u train.py --config configs/joint/joint_2gpu_b232_cpu18_b24_maxperf_seed43.yaml

echo "============================================================================"
echo ">>> [2/2] Running V5 MAXPERF Baseline with SEED 44 (20 Epochs)"
echo "Config: configs/joint/joint_2gpu_b232_cpu18_b24_maxperf_seed44.yaml"
echo "============================================================================"
python -u train.py --config configs/joint/joint_2gpu_b232_cpu18_b24_maxperf_seed44.yaml

echo "============================================================================"
echo ">>> All Seeds Finished! Running Statistical Aggregation..."
echo "============================================================================"
python3 scripts/aggregate_v5_stats.py
