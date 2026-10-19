# Lệnh chạy paired workload diagnostic A/B

Chạy trên **lab01**, từ terminal của bạn. Mỗi lần gọi tạo thư mục kết quả mới, không ghi đè run cũ.

```bash
cd /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI
mkdir -p logs
PAIR_LOG="logs/paired_workload_$(TZ=Asia/Ho_Chi_Minh date +%Y%m%d_%H%M%S).log"

nohup flock -n /tmp/heterovit_paired_workload_cluster.lock \
  /home/icip/Ha/HeteroViT-Project/mpi_cluster/python_by_host.sh \
  tools/run_paired_workload_diagnostic.py --phase all \
  > "$PAIR_LOG" 2>&1 < /dev/null &

echo "Launcher log: $PAIR_LOG"
```

`flock` ngăn chạy đồng thời hai lượt được mở bằng lệnh này. `nohup` cho phép job tiếp tục khi đóng terminal/SSH. Chỉ mở một lượt trên cụm; các job training khác không dùng cùng lock vẫn có thể cạnh tranh tài nguyên.

## Theo dõi trong cùng terminal

```bash
tail -f "$PAIR_LOG"
```

Nếu mở terminal mới:

```bash
cd /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI
ls -t logs/paired_workload_*.log
# Chọn đúng file vừa chạy rồi:
tail -f logs/paired_workload_YYYYMMDD_HHMMSS.log
```

Dòng `PAIR_ROOT:` trong launcher log cho biết thư mục kết quả chính xác. Để xem tiến độ từng epoch, dùng `tail -f <PAIR_ROOT>/A_control.log`, sau đó `tail -f <PAIR_ROOT>/B_mild_rebalance.log`; hai full logs xuất hiện khi run tương ứng bắt đầu.

## Lệnh thực hiện những gì?

1. Kiểm tra A/B và baseline: 40 epochs, seed42, H40, FP32, AdamW, LR0.001, global batch300; chỉ đổi allocation và tên output. Cùng telemetry ở cả A/B.
2. Snapshot source riêng; kiểm tra môi trường từng rank khớp baseline, đủ2GPU trên rank0, dữ liệu và source có trên5máy.
3. Smoke 120steps cho control không telemetry, control có telemetry và mild rebalance; kiểm tra clock/round logs. Dừng trước full nếu median compute overhead quan sát được vượt5% trên bất kỳ rank nào.
4. Chạy tuần tự **A=[266,6,6,6,16], 40epochs** rồi **B=[266,6,5,5,18], 40epochs**. Mỗi run khởi tạo mới từ seed42; không resume checkpoint dở dang.
5. Thu logs5ranks, align clock và sinh báo cáo Markdown/JSON/CSV.

Hai full runs thường cần khoảng80–90phút theo các runs cũ, chưa cộng startup, smoke và thu logs. Đây là ước lượng, không phải giới hạn thời gian.

## Kết quả

Trong thư mục `PAIR_ROOT`:

- `config_check.json`: xác nhận cấu hình và tổng batch300.
- `status.txt`: prepared/running/completed hoặc lỗi.
- `smoke_check.json`: kiểm tra telemetry và overhead.
- `A_control.log`, `B_mild_rebalance.log`: logs của hai full runs.
- `ranks/A_control/rank0..4/`, `ranks/B_mild_rebalance/rank0..4/`: raw CSV và clock calibration.
- `A_control_rounds.csv`, `B_mild_rebalance_rounds.csv`: readiness spread/critical worker từng round.
- `A_control_analysis.json`, `B_mild_rebalance_analysis.json`, `comparison.json`.
- **`paired_diagnostic_report_vi.md`**: báo cáo A/B.

Nếu cần tái tạo báo cáo sau khi cả hai full runs đã hoàn tất:

```bash
/home/icip/Ha/HeteroViT-Project/mpi_cluster/python_by_host.sh \
  tools/paired_workload_analysis.py /duong/dan/PAIR_ROOT
```

Lượt chạy trước đã dừng theo yêu cầu người dùng. Thư mục kết quả dở dang `paired_workload_seed42_20261019_003505_b9td6vr6` và source staging tạm trên lab01 đã được xóa theo yêu cầu; chạy mới A → B bằng lệnh trên.
