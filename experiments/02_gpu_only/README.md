# Experiment Phase 02: GPU-Only Baseline (1 V100)

## Mục tiêu
Đánh giá throughput và convergence baseline của 1 GPU NVIDIA Tesla V100 trên cùng node và cùng dataset CIFAR-10.

## Quy tắc thực nghiệm
1. Chỉ kích hoạt GPU 0 bằng `CUDA_VISIBLE_DEVICES=0`.
2. Giữ nguyên mọi siêu tham số so với CPU-only:
   - Cùng ViT-Tiny architecture
   - Cùng data split & augmentation
   - Cùng seed (42)
   - Cùng batch size (128)
   - Cùng optimizer (AdamW, lr=0.001, weight_decay=0.0001)
   - Cùng số epoch (100)
3. Làm mốc tham chiếu (speedup baseline) cho các giai đoạn tiếp theo (Heterogeneous CPU + 1 V100).
