#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Local SGD Sweep Launcher across H in {1, 4, 8, 16}
# ==============================================================================
# Executes sequential training runs for different local synchronization periods H.
# Each experiment logs to its own dedicated file and results directory.
#
# Usage:
#   ./scripts/sweep_local_sgd_h.sh               # Runs full 30 epochs for H in 1, 4, 8, 16
#   ./scripts/sweep_local_sgd_h.sh --epochs 10   # Overrides epochs to 10
#   ./scripts/sweep_local_sgd_h.sh --max-steps 20# Fast smoke test sweep (20 steps each)
#   nohup ./scripts/sweep_local_sgd_h.sh > sweep_local_sgd.log 2>&1 &
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_300.yaml"

# Target H values to sweep
H_VALUES=(1 4 8 16)
AVG_POLICY="sample_weighted"

echo "================================================================================"
echo " HeteroViT-MPI: Local SGD H-Sweep (H in ${H_VALUES[*]})"
echo "================================================================================"
echo " Config File  : $CONFIG"
echo " Target H's   : ${H_VALUES[*]}"
echo " Avg Policy   : $AVG_POLICY (Sample-Weighted for Heterogeneous Cluster)"
echo " Start Time   : $(date '+%Y-%m-%d %H:%M:%S')"
echo " Extra Args   : $*"
echo "================================================================================"

for H in "${H_VALUES[@]}"; do
  H_STR=$(printf "%02d" "$H")
  EXP_NAME="local_sgd_weighted_h${H_STR}_300"
  LOG_FILE="$PROJECT_ROOT/HeteroViT-MPI/train_${EXP_NAME}.log"

  echo ""
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo ">>> [STARTING EXPERIMENT] H = $H | Policy: $AVG_POLICY | Name: $EXP_NAME"
  echo ">>> Log File: $LOG_FILE"
  echo ">>> Time    : $(date '+%Y-%m-%d %H:%M:%S')"
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"

  set +e
  "$MPI_CLUSTER_DIR/run_5nodes.sh" \
    --sync \
    --config "$CONFIG" \
    --name "$EXP_NAME" \
    --local-sgd-h "$H" \
    --avg-policy "$AVG_POLICY" \
    "$@" 2>&1 | tee "$LOG_FILE"
  
  EXIT_CODE=${PIPESTATUS[0]}
  set -e

  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "================================================================================"
    echo ">>> [ERROR] Experiment with H=$H failed with exit code $EXIT_CODE."
    echo ">>> Check log: $LOG_FILE"
    echo "================================================================================"
    exit "$EXIT_CODE"
  fi

  echo "================================================================================"
  echo ">>> [COMPLETED] H = $H finished successfully at $(date '+%Y-%m-%d %H:%M:%S')"
  echo "================================================================================"
  
  # Brief cooldown between runs to allow OS network sockets/processes to recycle
  sleep 5
done

echo ""
echo "================================================================================"
echo " HeteroViT-MPI: Local SGD Sweep Complete!"
echo " Finished All H in (${H_VALUES[*]})"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
