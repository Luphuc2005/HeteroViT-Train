#!/usr/bin/env bash
# ==============================================================================
# HETEROVIT-MPI: PHASE 1 DISTRIBUTED HETEROGENEOUS SCHEDULER PROFILING & VALIDATION
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/../mpi_cluster" && pwd)"

HOSTFILE_5NODES="$MPI_CLUSTER_DIR/hostfile"
PYTHON_BY_HOST="$MPI_CLUSTER_DIR/python_by_host.sh"

cd "$PROJECT_ROOT"

echo "================================================================================"
echo " HETEROVIT-MPI: PHASE 1 PROFILING & VALIDATION SUITE"
echo "================================================================================"

MODE="${1:-all}"

case "$MODE" in
  network)
    echo "[*] Running Network Profiler (All-Pairs Latency/Bandwidth & Collective AllReduce)..."
    mpirun -x PYTHONNOUSERSITE=1 \
      --mca btl_tcp_if_include 192.168.1.0/24 \
      --mca oob_tcp_if_include 192.168.1.0/24 \
      -np 5 \
      --hostfile "$HOSTFILE_5NODES" \
      --map-by ppr:1:node \
      "$PYTHON_BY_HOST" -m src.scheduler.profiling.network_profiler
    ;;

  compute)
    echo "[*] Running Compute Profiler across all nodes in parallel (ViT-Tiny forward+backward)..."
    mpirun -x PYTHONNOUSERSITE=1 \
      --mca btl_tcp_if_include 192.168.1.0/24 \
      --mca oob_tcp_if_include 192.168.1.0/24 \
      -np 5 \
      --hostfile "$HOSTFILE_5NODES" \
      --map-by ppr:1:node \
      "$PYTHON_BY_HOST" -m src.scheduler.profiling.compute_profiler
    ;;

  validate)
    echo "[*] Running Offline Validation benchmarks on cluster..."
    python3 scripts/run_offline_validation.py
    ;;

  parse-only)
    echo "[*] Parsing existing benchmark metrics and generating validation table..."
    python3 scripts/run_offline_validation.py --parse-only
    ;;

  all)
    echo "[*] 1/3 Running Network Profiler..."
    mpirun -x PYTHONNOUSERSITE=1 \
      --mca btl_tcp_if_include 192.168.1.0/24 \
      --mca oob_tcp_if_include 192.168.1.0/24 \
      -np 5 \
      --hostfile "$HOSTFILE_5NODES" \
      --map-by ppr:1:node \
      "$PYTHON_BY_HOST" -m src.scheduler.profiling.network_profiler

    echo "[*] 2/3 Running Compute Profiler..."
    mpirun -x PYTHONNOUSERSITE=1 \
      --mca btl_tcp_if_include 192.168.1.0/24 \
      --mca oob_tcp_if_include 192.168.1.0/24 \
      -np 5 \
      --hostfile "$HOSTFILE_5NODES" \
      --map-by ppr:1:node \
      "$PYTHON_BY_HOST" -m src.scheduler.profiling.compute_profiler

    echo "[*] 3/3 Running Offline Validation..."
    python3 scripts/run_offline_validation.py
    ;;

  *)
    echo "Usage: $0 {network|compute|validate|parse-only|all}"
    exit 1
    ;;
esac

echo "[*] Phase 1 suite finished successfully."
