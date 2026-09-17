#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo "Starting Full GPU Benchmark Suite with Constrained CPU Cores"
echo "Batch Size: 256 | Epochs: 20"
echo "2 Cores -> 4 Cores -> 6 Cores -> 8 Cores"
echo "=========================================================="

echo ""
echo ">>> [1/4] Running GPU benchmark with 2 CPU cores..."
bash scripts/run_gpu_cpu2.sh

echo ""
echo ">>> [2/4] Running GPU benchmark with 4 CPU cores..."
bash scripts/run_gpu_cpu4.sh

echo ""
echo ">>> [3/4] Running GPU benchmark with 6 CPU cores..."
bash scripts/run_gpu_cpu6.sh

echo ""
echo ">>> [4/4] Running GPU benchmark with 8 CPU cores..."
bash scripts/run_gpu_cpu8.sh

echo ""
echo "=========================================================="
echo "ALL GPU CPU-CONSTRAINED BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "=========================================================="
