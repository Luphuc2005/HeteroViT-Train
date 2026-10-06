#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Master Experiment Suite (30 Epochs)
# Suite: AllReduce Baseline -> Local SGD H=4 -> Local SGD H=8 -> Local SGD H=16
# ==============================================================================
# Evaluates communication-computation tradeoff on 5-node heterogeneous cluster:
# - Wall-clock training duration (30 epochs)
# - Cluster throughput (samples / sec)
# - Total communication rounds & transferred MB
# - Best Validation Accuracy & Final Test Accuracy
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXEC="/home/icip/Ha/myenv/bin/python"

echo "================================================================================"
echo " HeteroViT-MPI: Master Distributed Training Benchmark Suite (30 Epochs)"
echo " Suite: AllReduce Baseline -> Local SGD (H in {4, 8, 16})"
echo " Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

# ------------------------------------------------------------------------------
# 1. AllReduce Baseline (Zero-Idle Balancer, Global Batch 300)
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [STAGE 1/4] Starting AllReduce Baseline (Sync every step, 30 Epochs)..."
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$SCRIPT_DIR/run_clean_allreduce_300.sh" \
  --name "baseline_allreduce_300" \
  --epochs 30 "$@"

echo ">>> Stage 1 Finished at $(date '+%Y-%m-%d %H:%M:%S')"
sleep 5

# ------------------------------------------------------------------------------
# 2. Local SGD H = 4 (Sample-Weighted Model Averaging every 4 steps)
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [STAGE 2/4] Starting Local SGD H = 4 (sample_weighted, 30 Epochs)..."
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$SCRIPT_DIR/run_local_sgd_300.sh" \
  --name "local_sgd_weighted_h04_300" \
  --local-sgd-h 4 \
  --avg-policy "sample_weighted" \
  --epochs 30 "$@"

echo ">>> Stage 2 Finished at $(date '+%Y-%m-%d %H:%M:%S')"
sleep 5

# ------------------------------------------------------------------------------
# 3. Local SGD H = 8 (Sample-Weighted Model Averaging every 8 steps)
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [STAGE 3/4] Starting Local SGD H = 8 (sample_weighted, 30 Epochs)..."
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$SCRIPT_DIR/run_local_sgd_300.sh" \
  --name "local_sgd_weighted_h08_300" \
  --local-sgd-h 8 \
  --avg-policy "sample_weighted" \
  --epochs 30 "$@"

echo ">>> Stage 3 Finished at $(date '+%Y-%m-%d %H:%M:%S')"
sleep 5

# ------------------------------------------------------------------------------
# 4. Local SGD H = 16 (Sample-Weighted Model Averaging every 16 steps)
# ------------------------------------------------------------------------------
echo ""
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>> [STAGE 4/4] Starting Local SGD H = 16 (sample_weighted, 30 Epochs)..."
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
"$SCRIPT_DIR/run_local_sgd_300.sh" \
  --name "local_sgd_weighted_h16_300" \
  --local-sgd-h 16 \
  --avg-policy "sample_weighted" \
  --epochs 30 "$@"

echo ">>> Stage 4 Finished at $(date '+%Y-%m-%d %H:%M:%S')"
sleep 5

# ------------------------------------------------------------------------------
# 5. Output Final Comparative Summary Table
# ------------------------------------------------------------------------------
echo ""
echo "================================================================================"
echo ">>> [FINAL BENCHMARK COMPARISON TABLE]"
echo "================================================================================"
"$PYTHON_EXEC" "$SCRIPT_DIR/compare_experiments.py" --pattern "*_300_*"

echo ""
echo "================================================================================"
echo " Master Suite Completed Successfully at $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

