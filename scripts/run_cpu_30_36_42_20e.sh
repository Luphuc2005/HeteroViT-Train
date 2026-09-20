#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "================================================================================"
echo ">>> Starting Automated CPU Benchmark Suite: 30 -> 36 -> 42 Threads"
echo ">>> Target Node: lab1 (48 Threads total)"
echo ">>> Epochs per run: 20"
echo ">>> Start Time: $(date)"
echo "================================================================================"

TOTAL_START=$(date +%s)

run_benchmark() {
  local name="$1"
  local script="$2"
  echo ""
  echo "--------------------------------------------------------------------------------"
  echo ">>> [START] Benchmark: $name at $(date)"
  echo "--------------------------------------------------------------------------------"
  local start_time
  start_time=$(date +%s)

  bash "$script"

  local end_time
  end_time=$(date +%s)
  local elapsed=$((end_time - start_time))
  echo ">>> [DONE] Benchmark $name completed in ${elapsed}s ($((elapsed / 60))m $((elapsed % 60))s)."
}

run_benchmark "30 Threads" "scripts/run_cpu_30_20e.sh"
run_benchmark "36 Threads" "scripts/run_cpu_36_20e.sh"
run_benchmark "42 Threads" "scripts/run_cpu_42_20e.sh"

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo ""
echo "================================================================================"
echo ">>> ALL 3 BENCHMARKS (30, 36, 42 THREADS) COMPLETED SUCCESSFULLY!"
echo ">>> Total Duration: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60))m $((TOTAL_ELAPSED % 60))s)"
echo ">>> Finished Time: $(date)"
echo "================================================================================"

