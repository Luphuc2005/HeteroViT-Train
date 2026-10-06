#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: [2/4] Local SGD H = 4 (20 Epochs, sample_weighted, Batch 300)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_300.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/train_02_local_sgd_h04_20e.log"

EPOCHS="${1:-20}"

echo "================================================================================"
echo ">>> [STARTING] 2/4 Local SGD H = 4 (Target Epochs: $EPOCHS)"
echo ">>> Log File: $LOG_FILE"
echo ">>> Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --name "local_sgd_h04_300_20e" \
  --local-sgd-h 4 \
  --avg-policy "sample_weighted" \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"
