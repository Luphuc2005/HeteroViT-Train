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

echo "=========================================================="
echo "Starting 2-GPU Benchmark Suite: 2x NVIDIA GeForce GTX TITAN Z"
echo "CUDA_VISIBLE_DEVICES: '$CUDA_VISIBLE_DEVICES'"
echo "Benchmarks: Global Batch 128 -> Global Batch 256 (20 epochs)"
echo "=========================================================="

echo ""
echo ">>> [1/2] Đang chạy Global Batch 128 (mỗi GPU nhận 64 samples)..."
python train.py --config configs/gpu/gpu_2titanz_b128_20e.yaml

echo ""
echo ">>> [2/2] Đang chạy Global Batch 256 (mỗi GPU nhận 128 samples)..."
python train.py --config configs/gpu/gpu_2titanz_b256_20e.yaml

echo ""
echo "=========================================================="
echo "HOÀN THÀNH CẢ 2 BÀI BENCHMARK 2 GPU TITAN Z!"
echo "Kết quả Batch 128 lưu tại: results/gpu_multi_benchmarks/b128/"
echo "Kết quả Batch 256 lưu tại: results/gpu_multi_benchmarks/b256/"
echo "=========================================================="
