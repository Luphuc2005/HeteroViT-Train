#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Suite H=80, H=120, H=150 (40 Epochs, Sample Weighted, Seed 42)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EPOCHS="${1:-40}"

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
  echo ">>> [$step_num/3] STARTING: $step_name ($EPOCHS Epochs, Seed 42)"
  echo ">>> Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "================================================================================"
  
  cleanup_orphans
  
  if bash "$SCRIPT_DIR/$step_script" "$EPOCHS"; then
    echo ">>> [$step_num/3] Completed successfully: $step_name at $(date '+%Y-%m-%d %H:%M:%S')"
  else
    echo "!!! [$step_num/3] FAILED: $step_name (Exit code $?)" >&2
    exit 1
  fi
  
  sleep 5
}

echo "================================================================================"
echo " HeteroViT-MPI: Starting H=80, H=120, H=150 Suite (40 Epochs, Seed 42)"
echo " Target Epochs: $EPOCHS per experiment"
echo " Started at   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Step 1: Local SGD H=80 (sample_weighted, 40 Epochs)
run_step "1" "Local SGD H = 80 (Sample Weighted)" "run_14_local_sgd_h80_40e.sh"

# Step 2: Local SGD H=120 (sample_weighted, 40 Epochs)
run_step "2" "Local SGD H = 120 (Sample Weighted)" "run_15_local_sgd_h120_40e.sh"

# Step 3: Local SGD H=150 (sample_weighted, 40 Epochs)
run_step "3" "Local SGD H = 150 (Sample Weighted)" "run_16_local_sgd_h150_40e.sh"

cleanup_orphans

echo ""
echo "================================================================================"
echo " HeteroViT-MPI: Suite Completed Successfully!"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

python3 "$SCRIPT_DIR/aggregate_benchmark_results.py"

