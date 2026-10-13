#!/usr/bin/env bash
# Repeat only FP32 and FP16 delta AllGather with seed 43 (40 epochs by default).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/sweep_fp16_comparison.sh" "$@" \
  --seed 43 --methods fp32,fp16_delta_allgather
