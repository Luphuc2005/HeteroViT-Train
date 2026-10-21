#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_h40_fp16_delta_allgather_cluster5_rebalance_600_40e.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/train_25_cluster5_b600_h40_fp16_delta.log"

EPOCHS="${1:-40}"

echo "================================================================================"
echo ">>> [STARTING] Cluster5 Batch 600 H=40 FP16 Delta AllGather (40 Epochs)"
echo ">>> 50% Payload Compression + 50% Reduced Sync Frequency"
echo ">>> Log File: $LOG_FILE"
echo ">>> Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"

