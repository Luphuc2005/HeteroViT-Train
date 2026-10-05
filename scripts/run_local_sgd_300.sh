#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# HeteroViT-MPI: Local SGD / Periodic Model Averaging Launcher
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MPI_CLUSTER_DIR="$(cd "$PROJECT_ROOT/mpi_cluster" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/mpi/local_sgd_300.yaml"

exec "$MPI_CLUSTER_DIR/run_5nodes.sh" --sync --config "$CONFIG" "$@"
