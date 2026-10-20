#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1

echo "=========================================================="
echo ">>> [BENCHMARK] Pure 2-GPU Baseline (Batch 600, 40 Epochs)"
echo ">>> Strategy: tf.distribute.MirroredStrategy across 2 Titan Z GPUs"
echo "=========================================================="

python train.py --config configs/gpu/gpu_2titanz_b600_40e.yaml "$@"
