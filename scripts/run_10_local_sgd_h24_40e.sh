#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Local SGD H = 24 (40 Epochs, Seed 42, Batch 300)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_h24_300_40e.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/train_10_local_sgd_h24_40e.log"

EPOCHS="${1:-40}"

echo "================================================================================"
echo ">>> [STARTING] Local SGD H = 24 (40 Epochs, Seed 42)"
echo ">>> Log File: $LOG_FILE"
echo ">>> Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --name "local_sgd_h24_300_40e" \
  --local-sgd-h 24 \
  --avg-policy "sample_weighted" \
  --seed 42 \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"
