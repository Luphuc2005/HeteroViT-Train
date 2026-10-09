#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Drift-Adaptive-H Local SGD (40 Epochs, Seed 42, Batch 300)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_adaptive_h_300_40e.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/logs/adaptive_40e/train_17_local_sgd_drift_adaptive_h_40e.log"
mkdir -p "$(dirname "$LOG_FILE")"

EPOCHS="${1:-40}"

echo "================================================================================"
echo ">>> [STARTING] Drift-Adaptive-H Local SGD (40 Epochs, Seed 42)"
echo ">>> Candidates: [40, 80, 120, 150] | Initial H: 40"
echo ">>> Thresholds: tau_low=0.02, tau_high=0.05"
echo ">>> Log File  : $LOG_FILE"
echo ">>> Time      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --name "local_sgd_adaptive_h_300_40e" \
  --seed 42 \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"

