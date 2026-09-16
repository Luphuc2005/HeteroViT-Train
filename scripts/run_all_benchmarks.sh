#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================================="
echo "Starting Full CPU Benchmark Suite (20 epochs each)"
echo "18 Cores -> 24 Cores -> 48 Threads"
echo "=========================================================="

bash scripts/run_cpu_18_20e.sh
bash scripts/run_cpu_24_20e.sh
bash scripts/run_cpu_48t_20e.sh

echo "=========================================================="
echo "ALL CPU BENCHMARKS COMPLETED SUCCESSFULLY!"
echo "=========================================================="
