# GPU-only sweep seed43 → seed44

Chạy trên lab01: một MPI rank dùng hai GPU qua MirroredStrategy; global batch 300,
150 samples/GPU/step. Mỗi seed khởi tạo mới và chạy 40 epochs. Config giống
GPU-only seed42 ngoài seed và tên experiment: FP32, H40, AdamW, LR 0,001,
model/dataset/augmentation/evaluation và diagnostic logging giữ nguyên.

## Một lệnh chạy nền

```bash
cd /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI && mkdir -p logs && nohup ../mpi_cluster/python_by_host.sh tools/run_gpu_only_seed43_44_sweep.py > "logs/gpu_only_seed43_44_$(date +%Y%m%d_%H%M%S).log" 2>&1 < /dev/null &
```

Không bọc thêm `flock`: runner tự giữ lock `/tmp/heterovit_paired_workload_cluster.lock`
suốt cả sweep để tránh chồng với những lệnh experiment dùng cùng lock. Không
chạy hai seeds đồng thời. Không resume, không chạy lại seed42 và không chạy smoke
A/B. Nếu một run hoặc kiểm tra kết quả thất bại, runner dừng và giữ kết quả dở
trong thư mục riêng, không chạy seed tiếp theo.

## Theo dõi

```bash
tail -f "$(ls -t /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/logs/gpu_only_seed43_44_*.log | head -n 1)"
```

Log này hiển thị cả tiến trình sweep và từng epoch của training. `Ctrl+C` chỉ
thoát xem log. Dòng `SWEEP_ROOT` cho biết thư mục kết quả duy nhất của lượt chạy.

Trong `SWEEP_ROOT` có `seed43.log`, `seed44.log`, `runs/` chứa checkpoint và CSV
từng seed; `summary.csv`, `summary.json` được cập nhật sau mỗi seed thành công;
`status.txt` ghi tiến độ hoặc lỗi. `source/`, `source_manifest.json` và
`config_check.json` giữ snapshot và config đã kiểm tra. Hai seeds dùng chung
snapshot source, không ghi đè run cũ.

Summary kiểm tra đủ 40 epochs, 6.000 optimizer steps, 160 syncs và 1.800.000
processed samples/run; ghi best/final validation, test từ best-validation
checkpoint, training-loop time, epoch đầu đạt validation 65% và elapsed tới mốc
này. Accuracy CSV có dạng tỷ lệ 0–1; summary chuyển thành phần trăm.

Chỉ kiểm tra config, không chạy training:

```bash
../mpi_cluster/python_by_host.sh tools/run_gpu_only_seed43_44_sweep.py --check-only
```

Đây là phép bổ sung seeds cho baseline GPU-only. Cùng batch với cụm năm máy
không có nghĩa cùng training trajectory; phải báo cả thời gian và chất lượng.
