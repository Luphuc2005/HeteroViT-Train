#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MODE="${1:-observe}"
case "$MODE" in
    observe)
        CONFIG="configs/joint/joint_2gpu_b232_cpu18_v6_observe.yaml"
        ;;
    dynamic)
        CONFIG="configs/joint/joint_2gpu_b232_cpu18_v6_dynamic.yaml"
        ;;
    *)
        echo "Usage: bash scripts/run_joint_v6.sh [observe|dynamic]" >&2
        exit 2
        ;;
esac

echo "============================================================================"
echo ">>> V6 Dynamic CPU-GPU Workload Scheduler ($MODE)"
echo "Config: $CONFIG"
echo "============================================================================"

export PYTHONUNBUFFERED=1
python -u train.py --config "$CONFIG"
