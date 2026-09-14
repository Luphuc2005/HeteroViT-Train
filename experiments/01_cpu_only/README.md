# Experiment Phase 01: CPU-Only Baseline

## Mục tiêu
Đánh giá năng lực tính toán và throughput (samples/sec, epoch_time) của ViT-Tiny trên các mức CPU cores khác nhau (6, 12, 18, 24 physical cores) trên cùng 1 GPU node.

## Quy tắc thực nghiệm
1. Vô hiệu hóa GPU bằng `CUDA_VISIBLE_DEVICES=""`.
2. Thiết lập thread affinity chuẩn bằng `taskset`:
   - 6 cores: `taskset -c 0-5`
   - 12 cores: `taskset -c 0-11`
   - 18 cores: `taskset -c 0-17`
   - 24 cores: `taskset -c 0-23`
3. Cùng seed (`42`), batch size (`128`), optimizer (`AdamW`), ViT-Tiny kiến trúc chuẩn.
