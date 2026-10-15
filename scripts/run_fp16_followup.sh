#!/usr/bin/env bash
# Defaults to the bounded diagnostic; full training is a separate phase.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
exec "$PROJECT_ROOT/mpi_cluster/python_by_host.sh" \
  "$SCRIPT_DIR/../tools/run_fp16_followup.py" "$@"
