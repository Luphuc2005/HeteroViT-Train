#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: V2 Online Drift-Dynamics Adaptive-H (40 Epochs, Seed 42, Batch 300)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_adaptive_v2_odd_300_40e.yaml"
LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/logs/adaptive_40e/train_19_local_sgd_adaptive_v2_odd_40e.log"
mkdir -p "$(dirname "$LOG_FILE")"

EPOCHS="${1:-40}"

echo "================================================================================"
echo ">>> [STARTING] V2 ODD-H Local SGD (40 Epochs, Seed 42, Batch 300)"
echo ">>> Candidates: [40, 80, 120, 150] | Initial H: 40 | Budget: 0.05"
echo ">>> RLS Forgetting: lambda=0.98 | Uncertainty: kappa=1.0"
echo ">>> Log File  : $LOG_FILE"
echo ">>> Time      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG" \
  --name "adaptive_v2_online_dynamics_300_40e_seed42" \
  --seed 42 \
  --epochs "$EPOCHS" "${@:2}" 2>&1 | tee "$LOG_FILE"

