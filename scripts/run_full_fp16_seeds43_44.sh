#!/usr/bin/env bash
# Run all five H=40 methods, 40 epochs each, for seed 43 then seed 44.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRA_ARGS=()
case "${1:-}" in
  "") ;;
  --dry-run) EXTRA_ARGS=(--dry-run) ;;
  -h|--help)
    echo "Usage: bash $0 [--dry-run]"
    echo "Runs all five methods sequentially for seeds 43 and 44 (40 epochs, H=40)."
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (($# > 1)); then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi

METHODS=fp32,fp16_whole_allgather,fp16_whole_allreduce,fp16_delta_allgather,fp16_delta_allreduce
for seed in 43 44; do
  echo "Starting full five-method sweep: seed=$seed, epochs=40, H=40"
  bash "$SCRIPT_DIR/sweep_fp16_comparison.sh" \
    --seed "$seed" --epochs 40 --methods "$METHODS" "${EXTRA_ARGS[@]}"
done
echo "Completed both seeds."
