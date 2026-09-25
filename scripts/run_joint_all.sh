#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

chmod +x scripts/run_joint_cpu18_b32.sh scripts/run_joint_cpu18_b48.sh

TARGET=${1:-all}

case "$TARGET" in
    32)
        bash scripts/run_joint_cpu18_b32.sh
        ;;
    48)
        bash scripts/run_joint_cpu18_b48.sh
        ;;
    all)
        echo "============================================================================"
        echo "STARTING FULL JOINT HETEROGENEOUS BENCHMARKS: CPU BATCH 32 -> CPU BATCH 48"
        echo "============================================================================"
        bash scripts/run_joint_cpu18_b32.sh
        echo ""
        echo "Sleeping 10s between runs..."
        sleep 10
        bash scripts/run_joint_cpu18_b48.sh
        echo ""
        echo "============================================================================"
        echo "ALL JOINT HETEROGENEOUS BENCHMARKS COMPLETED SUCCESSFULLY!"
        echo "============================================================================"
        ;;
    *)
        echo "Usage: bash scripts/run_joint_all.sh [32|48|all]"
        exit 1
        ;;
esac
