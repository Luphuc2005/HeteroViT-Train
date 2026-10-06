#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Extended Local SGD Suite (H=24, H=32)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EPOCHS="${1:-20}"

cleanup_orphans() {
  echo ">>> [CLEANUP] Ensuring no orphan python/mpirun processes on cluster..."
  pkill -9 -f "train_mpi.py" 2>/dev/null || true
  pkill -9 -f "mpirun" 2>/dev/null || true
  for host in iciplab02 iciplab03 iciplab04 iciplab05; do
    ssh -o ConnectTimeout=3 -o BatchMode=yes "$host" "pkill -9 -f 'train_mpi.py'" 2>/dev/null || true
  done
  sleep 3
}

run_step() {
  local step_num="$1"
  local step_name="$2"
  local step_script="$3"
  
  echo ""
  echo "================================================================================"
  echo ">>> [$step_num/2] STARTING: $step_name ($EPOCHS Epochs)"
  echo ">>> Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "================================================================================"
  
  cleanup_orphans
  
  if bash "$SCRIPT_DIR/$step_script" "$EPOCHS"; then
    echo ">>> [$step_num/2] Completed successfully: $step_name at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "!!! [$step_num/2] FAILED: $step_name (Exit code $?)" >&2
    exit 1
  fi
  
  sleep 5
}

echo "================================================================================"
echo " HeteroViT-MPI: Starting Extended Local SGD Suite (H=24, H=32)"
echo " Target Epochs: $EPOCHS per experiment"
echo " Started at   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Run 1: Local SGD H=24
run_step "1" "Local SGD H = 24" "run_05_local_sgd_h24_20e.sh"

# Run 2: Local SGD H=32
run_step "2" "Local SGD H = 32" "run_06_local_sgd_h32_20e.sh"

cleanup_orphans

echo ""
echo "================================================================================"
echo " HeteroViT-MPI: Extended Benchmark Runs Completed Successfully!"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

python3 "$SCRIPT_DIR/aggregate_benchmark_results.py"
