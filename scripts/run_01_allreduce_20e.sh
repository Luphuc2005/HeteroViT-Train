#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: [1/4] Clean AllReduce Baseline (20 Epochs, Batch 300)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/auto_zero_idle_300.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/train_01_allreduce_20e.log"

EPOCHS="${1:-20}"

echo "================================================================================"
echo ">>> [STARTING] 1/4 Clean AllReduce Baseline (Target Epochs: $EPOCHS)"
echo ">>> Log File: $LOG_FILE"
echo ">>> Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --name "allreduce_baseline_300_20e" \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"
