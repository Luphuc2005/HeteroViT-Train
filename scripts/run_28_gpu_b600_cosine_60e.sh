#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MPI_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$MPI_DIR/configs/gpu/gpu_2titanz_b600_cosine_60e.yaml"
LOG_FILE="$MPI_DIR/train_gpu_b600_cosine_60e.log"

source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1

echo "================================================================================"
echo ">>> [STARTING] Pure 2-GPU Baseline Batch 600 Cosine LR (60 Epochs)"
echo ">>> Config   : $CONFIG"
echo ">>> Log File : $LOG_FILE"
echo ">>> Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

cd "$MPI_DIR"
python train.py --config "$CONFIG" "${@}" 2>&1 | tee "$LOG_FILE"
