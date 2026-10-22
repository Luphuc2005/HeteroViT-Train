#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_h20_fp32_cosine_cluster5_rebalance_600_60e.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/train_29_cluster5_b600_h20_cosine_60e.log"

source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

EPOCHS="${1:-60}"

echo "================================================================================"
echo ">>> [STARTING] Cluster5 Batch 600 H=20 FP32 with Cosine LR (60 Epochs)"
echo ">>> Rebalance [544,12,6,6,32] + Cosine Annealing (1e-3 -> 1e-5)"
echo ">>> Log File: $LOG_FILE"
echo ">>> Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"
