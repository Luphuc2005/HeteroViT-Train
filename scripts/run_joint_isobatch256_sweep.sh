#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

chmod +x scripts/run_joint_isobatch256_b24.sh scripts/run_joint_isobatch256_b26.sh

TARGET=${1:-all}

case "$TARGET" in
    24|b24|opt1)
        bash scripts/run_joint_isobatch256_b24.sh
        ;;
    26|b26|opt2)
        bash scripts/run_joint_isobatch256_b26.sh
        ;;
    all)
        echo "============================================================================"
        echo "STARTING FULL JOINT ISO-BATCH 256 WORKLOAD BALANCING SWEEP"
        echo "  [Run 1/2]: GPU Batch 232 (116/GPU) + CPU Batch 24 (18 Cores) -> Total 256"
        echo "  [Run 2/2]: GPU Batch 230 (115/GPU) + CPU Batch 26 (18 Cores) -> Total 256"
        echo "============================================================================"
        echo ""
        echo ">>> Executing [Run 1/2]: GPU 232 + CPU 24..."
        bash scripts/run_joint_isobatch256_b24.sh
        echo ""
        echo "============================================================================"
        echo ">>> [Run 1/2] COMPLETED! Cooling down & sleeping 15s before [Run 2/2]..."
        echo "============================================================================"
        sleep 15
        echo ""
        echo ">>> Executing [Run 2/2]: GPU 230 + CPU 26..."
        bash scripts/run_joint_isobatch256_b26.sh
        echo ""
        echo "============================================================================"
        echo "ALL JOINT ISO-BATCH 256 BENCHMARKS COMPLETED SUCCESSFULLY!"
        echo "Results saved in:"
        echo "  - ./results/joint_benchmarks/isobatch256_gpu232_cpu24/"
        echo "  - ./results/joint_benchmarks/isobatch256_gpu230_cpu26/"
        echo "============================================================================"
        ;;
    *)
        echo "Usage: bash scripts/run_joint_isobatch256_sweep.sh [24|26|all]"
        exit 1
        ;;
esac
