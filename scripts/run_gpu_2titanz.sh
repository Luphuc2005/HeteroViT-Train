#!/usr/bin/env bash
set -e

# Đảm bảo chạy từ thư mục gốc của project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Kích hoạt 2 GPU Titan Z (GPU 0 và GPU 1)
export CUDA_VISIBLE_DEVICES=0,1

# Tự động liên kết libdevice.10.bc cho XLA nếu có
if [ ! -f "libdevice.10.bc" ]; then
    for cand in \
        /usr/local/cuda/nvvm/libdevice/libdevice.10.bc \
        /usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc \
        /usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc \
        "$CONDA_PREFIX/nvvm/libdevice/libdevice.10.bc" \
        "$VIRTUAL_ENV/lib/python*/site-packages/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc"; do
        if [ -f "$cand" ]; then
            ln -sf "$cand" ./libdevice.10.bc 2>/dev/null || true
            break
        fi
    done
fi

TARGET=${1:-all}

run_1() {
    echo ""
    echo "=========================================================="
    echo ">>> [RUN 1/3] 2 GPU Titan Z | 2 CPU Cores | Global Batch: 128 (Baseline Scaling)"
    echo "=========================================================="
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    export TF_NUM_INTRAOP_THREADS=2
    export TF_NUM_INTEROP_THREADS=2
    taskset -c 0-1 python train.py --config configs/gpu/gpu_2titanz_b128_20e.yaml
}

run_2() {
    echo ""
    echo "=========================================================="
    echo ">>> [RUN 2/3] 2 GPU Titan Z | 2 CPU Cores | Global Batch: 256 (Xem 2 Cores đủ không)"
    echo "=========================================================="
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2
    export NUMEXPR_NUM_THREADS=2
    export TF_NUM_INTRAOP_THREADS=2
    export TF_NUM_INTEROP_THREADS=2
    taskset -c 0-1 python train.py --config configs/gpu/gpu_2titanz_b256_20e.yaml
}

run_3() {
    echo ""
    echo "=========================================================="
    echo ">>> [RUN 3/3] 2 GPU Titan Z | 4 CPU Cores | Global Batch: 256 (Kiểm tra Loader Bottleneck)"
    echo "=========================================================="
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    export TF_NUM_INTRAOP_THREADS=4
    export TF_NUM_INTEROP_THREADS=2
    taskset -c 0-3 python train.py --config configs/gpu/gpu_2titanz_c4_b256_20e.yaml
}

case "$TARGET" in
    1)
        run_1
        ;;
    2)
        run_2
        ;;
    3)
        run_3
        ;;
    all)
        run_1
        run_2
        run_3
        echo ""
        echo "=========================================================="
        echo "HOÀN TẤT CẢ 3 BENCHMARK 2 GPU TITAN Z THÀNH CÔNG!"
        echo "Kết quả lưu riêng tại:"
        echo " - Run 1: results/gpu_multi_benchmarks/run1_c2_b128/"
        echo " - Run 2: results/gpu_multi_benchmarks/run2_c2_b256/"
        echo " - Run 3: results/gpu_multi_benchmarks/run3_c4_b256/"
        echo "=========================================================="
        ;;
    *)
        echo "Lựa chọn không hợp lệ: '$TARGET'. Hãy dùng: 1, 2, 3, hoặc all."
        exit 1
        ;;
esac
