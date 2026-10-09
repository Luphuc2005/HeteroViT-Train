#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Suite Adaptive Local SGD V1, V2, V3 (40 Epochs, Seed 42)
# Runs sequentially: V1 CBA-H -> V2 ODD-H -> V3 PDCA-H
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
echo ">>> [SUITE] HeteroViT-MPI: Adaptive Local SGD Evaluation Suite (40 Epochs)"
echo ">>> Sequence: V1 (CBA-H) -> V2 (ODD-H) -> V3 (PDCA-H)"
echo ">>> Epochs  : $EPOCHS"
echo ">>> Cluster : 5 Nodes (GPU Rank 0, 4x CPU Nodes, 100 Mbps Ethernet)"
echo ">>> Start   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Step 1: V1 Consensus-Budget Adaptive-H
run_step "1" "V1 CBA-H (Consensus-Budget Adaptive-H)" "run_18_local_sgd_adaptive_v1_cba_40e.sh"

# Step 2: V2 Online Drift-Dynamics Adaptive-H
run_step "2" "V2 ODD-H (Online Drift-Dynamics Adaptive-H)" "run_19_local_sgd_adaptive_v2_odd_40e.sh"

# Step 3: V3 Primal-Dual Cost-Aware Adaptive-H
run_step "3" "V3 PDCA-H (Primal-Dual Cost-Aware Adaptive-H)" "run_20_local_sgd_adaptive_v3_pdca_40e.sh"

echo ""
echo "================================================================================"
echo ">>> [COMPLETE] Adaptive Suite V1, V2, V3 finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

