#!/usr/bin/env bash
set -e

# Ensure we run from project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# GPU baseline: 1 V100 GPU on single GPU node
export CUDA_VISIBLE_DEVICES=0

echo "=========================================================="
echo "Starting GPU benchmark: 1 V100 GPU"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "=========================================================="

CUDA_VISIBLE_DEVICES=0 python train.py --config configs/gpu/gpu_1v100.yaml
