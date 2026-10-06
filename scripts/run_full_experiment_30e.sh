#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Full 30-Epoch Benchmark Experiment
# ==============================================================================
# Pipeline:
#   1. Clean AllReduce Baseline (Zero-Idle, Global Batch 300)
#   2. Local SGD H = 4          (sample_weighted, Global Batch 300)
#   3. Local SGD H = 8          (sample_weighted, Global Batch 300)
#   4. Local SGD H = 16         (sample_weighted, Global Batch 300)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"

CONFIG_ALLREDUCE="$SCRIPT_DIR/../configs/mpi/auto_zero_idle_300.yaml"
CONFIG_LOCAL_SGD="$SCRIPT_DIR/../configs/mpi/local_sgd_300.yaml"

EPOCHS="${1:-30}"

echo "================================================================================"
echo " HeteroViT-MPI: Starting Full Benchmark Pipeline (Target Epochs: $EPOCHS)"
echo " Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. Clean AllReduce Baseline
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [1/4] Running Clean AllReduce Baseline (300 Global Batch)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG_ALLREDUCE" \
  --name "allreduce_baseline_300_30e" \
  --epochs "$EPOCHS" 2>&1 | tee "$PROJECT_ROOT/HeteroViT-MPI/train_allreduce_baseline_30e.log"

sleep 10

# ------------------------------------------------------------------------------
# 2. Local SGD H = 4
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [2/4] Running Local SGD (H = 4, sample_weighted)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG_LOCAL_SGD" \
  --name "local_sgd_h04_300_30e" \
  --local-sgd-h 4 \
  --avg-policy "sample_weighted" \
  --epochs "$EPOCHS" 2>&1 | tee "$PROJECT_ROOT/HeteroViT-MPI/train_local_sgd_h04_30e.log"

sleep 10

# ------------------------------------------------------------------------------
# 3. Local SGD H = 8
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [3/4] Running Local SGD (H = 8, sample_weighted)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG_LOCAL_SGD" \
  --name "local_sgd_h08_300_30e" \
  --local-sgd-h 8 \
  --avg-policy "sample_weighted" \
  --epochs "$EPOCHS" 2>&1 | tee "$PROJECT_ROOT/HeteroViT-MPI/train_local_sgd_h08_30e.log"

sleep 10

# ------------------------------------------------------------------------------
# 4. Local SGD H = 16
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [4/4] Running Local SGD (H = 16, sample_weighted)"
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$MPI_CLUSTER_DIR/run_5nodes.sh" \
  --sync \
  --config "$CONFIG_LOCAL_SGD" \
  --name "local_sgd_h16_300_30e" \
  --local-sgd-h 16 \
  --avg-policy "sample_weighted" \
  --epochs "$EPOCHS" 2>&1 | tee "$PROJECT_ROOT/HeteroViT-MPI/train_local_sgd_h16_30e.log"

echo ""
echo "================================================================================"
echo " HeteroViT-MPI: Full Benchmark Pipeline Finished Successfully!"
echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

python3 "$SCRIPT_DIR/aggregate_benchmark_results.py" || true

