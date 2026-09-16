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
3. Cùng seed (`42`), optimizer (`AdamW`), ViT-Tiny kiến trúc chuẩn.

## Pha 1.1: Core Scaling Benchmark (Batch Size 128 cố định)
- 6 cores: `scripts/run_cpu_06_20e.sh` (`configs/cpu/cpu_06cores_20e.yaml`)
- 12 cores: `scripts/run_cpu_12_20e.sh` (`configs/cpu/cpu_12cores_20e.yaml`)
- 18 cores: `scripts/run_cpu_18_20e.sh` (`configs/cpu/cpu_18cores_20e.yaml`)
- 24 cores: `scripts/run_cpu_24_20e.sh` (`configs/cpu/cpu_24cores_20e.yaml`)
- 24 cores / 48 threads (HT): `scripts/run_cpu_48t_20e.sh` (`configs/cpu/cpu_48t_20e.yaml`)

## Pha 1.2: Batch Size Sweep Benchmark (Cố định 24 Cores)
Khảo sát thông lượng tính toán (samples/sec), thời gian epoch và bộ nhớ RAM khi thay đổi kích thước batch trên 24 physical cores:
- Batch 64: `scripts/run_cpu_24_b64_20e.sh` (`configs/cpu/cpu_24cores_b64_20e.yaml`)
- Batch 128: `scripts/run_cpu_24_b128_20e.sh` (`configs/cpu/cpu_24cores_b128_20e.yaml`)
- Batch 256: `scripts/run_cpu_24_b256_20e.sh` (`configs/cpu/cpu_24cores_b256_20e.yaml`)
- Batch 512: `scripts/run_cpu_24_b512_20e.sh` (`configs/cpu/cpu_24cores_b512_20e.yaml`)
- Chạy tự động toàn bộ sweep: `scripts/run_batch_sweep_24cores.sh`

