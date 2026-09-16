#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo "Starting 24-Core CPU Batch Size Sweep (20 epochs each)"
echo "Batch Sizes: 64 -> 128 -> 256 -> 512"
echo "=========================================================="

echo ""
echo ">>> [1/4] Running Batch Size 64..."
bash scripts/run_cpu_24_b64_20e.sh

echo ""
echo ">>> [2/4] Running Batch Size 128..."
bash scripts/run_cpu_24_b128_20e.sh

echo ""
echo ">>> [3/4] Running Batch Size 256..."
bash scripts/run_cpu_24_b256_20e.sh

echo ""
echo ">>> [4/4] Running Batch Size 512..."
bash scripts/run_cpu_24_b512_20e.sh

echo ""
echo "=========================================================="
echo "ALL BATCH SIZE SWEEP RUNS COMPLETED SUCCESSFULLY!"
echo "=========================================================="
