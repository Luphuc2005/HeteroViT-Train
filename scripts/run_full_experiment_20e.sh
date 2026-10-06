#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Full 20-Epoch Benchmark Experiment Suite
# ==============================================================================
# Pipeline:
#   [1/4] Clean AllReduce Baseline (20 epochs) -> train_01_allreduce_20e.log
#   [2/4] Local SGD H = 4          (20 epochs) -> train_02_local_sgd_h04_20e.log
#   [3/4] Local SGD H = 8          (20 epochs) -> train_03_local_sgd_h08_20e.log
#   [4/4] Local SGD H = 16         (20 epochs) -> train_04_local_sgd_h16_20e.log
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCHS="${1:-20}"

echo "================================================================================"
echo " HeteroViT-MPI: Starting Full 20-Epoch Benchmark Suite"
echo " Target Epochs per Run: $EPOCHS"
echo " Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# 1. Clean AllReduce Baseline
echo ""
"$SCRIPT_DIR/run_01_allreduce_20e.sh" "$EPOCHS"
echo ">>> [1/4] Completed at $(date '+%Y-%m-%d %H:%M:%S'). Cooling down 10s..."
sleep 10

# 2. Local SGD H = 4
echo ""
"$SCRIPT_DIR/run_02_local_sgd_h04_20e.sh" "$EPOCHS"
echo ">>> [2/4] Completed at $(date '+%Y-%m-%d %H:%M:%S'). Cooling down 10s..."
sleep 10

# 3. Local SGD H = 8
echo ""
"$SCRIPT_DIR/run_03_local_sgd_h08_20e.sh" "$EPOCHS"
echo ">>> [3/4] Completed at $(date '+%Y-%m-%d %H:%M:%S'). Cooling down 10s..."
sleep 10

# 4. Local SGD H = 16
echo ""
"$SCRIPT_DIR/run_04_local_sgd_h16_20e.sh" "$EPOCHS"
echo ">>> [4/4] Completed at $(date '+%Y-%m-%d %H:%M:%S')."

echo ""
echo "================================================================================"
echo " HeteroViT-MPI: All 4 Benchmark Runs Completed Successfully!"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# Print aggregated summary
python3 "$SCRIPT_DIR/aggregate_benchmark_results.py" || true
