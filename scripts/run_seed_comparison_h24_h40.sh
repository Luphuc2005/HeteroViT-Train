#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Local SGD Seed Verification Suite (H=24 vs H=40, Seed 43)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EPOCHS="${1:-20}"

cleanup_orphans() {
  echo ">>> [CLEANUP] Ensuring no orphan python/mpirun processes on cluster..."
  pkill -9 -f "train_mpi.py" 2>/dev/null || true
  pkill -9 -f "mpirun" 2>/dev/null || true
  for host in 192.168.1.132 192.168.1.133 192.168.1.134 192.168.1.135; do
    ssh -o ConnectTimeout=3 -o BatchMode=yes icip@"$host" "pkill -9 -f 'train_mpi.py'; pkill -9 -f 'mpirun'" 2>/dev/null || true
  done
  sleep 3
}

run_step() {
  local step_num="$1"
  local step_name="$2"
  local step_script="$3"
  
  echo ""
  echo "================================================================================"
  echo ">>> [$step_num/2] STARTING: $step_name ($EPOCHS Epochs, Seed 43)"
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
echo " HeteroViT-MPI: Starting Seed Verification Suite (H=24 vs H=40 with Seed 43)"
echo " Target Epochs: $EPOCHS per experiment"
echo " Started at   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Run 1: Local SGD H=24 (Seed 43)
run_step "1" "Local SGD H = 24 (Seed 43)" "run_08_local_sgd_h24_seed43_20e.sh"

# Run 2: Local SGD H=40 (Seed 43)
run_step "2" "Local SGD H = 40 (Seed 43)" "run_09_local_sgd_h40_seed43_20e.sh"

cleanup_orphans

echo ""
echo "================================================================================"
echo " HeteroViT-MPI: Seed Verification Suite Completed Successfully!"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

python3 "$SCRIPT_DIR/compare_seeds_h24_h40.py"
