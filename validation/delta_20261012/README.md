# Kiểm chứng FP16 Delta trên cluster 5 node — 12/10/2026

Runtime preflight, transport regression và smoke training của cả hai mode đều
**PASS**. Đây là kết quả kiểm tra chức năng, chưa phải benchmark 40 epochs.

| Node | Python | TensorFlow | NumPy | mpi4py | OpenMPI |
|---|---|---|---|---|---|
| lab01 | 3.8.10 | 2.10.1 | 1.23.5 | 4.1.1 | 4.0.3 |
| lab02 | 3.8.10 | 2.10.0 | 1.23.5 | 4.1.1 | 4.0.2 |
| lab03 | 3.8.10 | 2.10.0 | 1.23.5 | 4.1.1 | 4.0.2 |
| lab04 | 3.8.10 | 2.10.0 | 1.23.5 | 4.1.1 | 4.0.2 |
| lab05 | 3.8.10 | 2.10.1 | 1.23.5 | 4.1.1 | 4.0.2 |

Đây là runtime của `python_by_host.sh`, khác bảng phiên bản Python hệ thống.
TensorFlow import thành công; không cài lại package hay thay đổi môi trường.
Cả 5 ranks có cùng ordered model layout 2.693.578 tham số, compute FP32 và
`keras.optimizers.optimizer_experimental.adamw.AdamW`.

Regression: **12 tests/rank trên 5 máy thật**, gồm FP32, FP16 whole-weight
AllGather/AllReduce và delta. File baseline communicator và các YAML baseline
hiện có không thay đổi.

| Kiểm chứng | Delta AllGather | Delta AllReduce |
|---|---|---|
| Local steps / H / số sync | 120 / 40 / 3 | 120 / 40 / 3 |
| Optimizer iteration trên mọi rank | 120 | 120 |
| Model + optimizer state hữu hạn FP32 | PASS | PASS |
| Model cuối khớp anchor, SHA256 giống trên 5 ranks | PASS | PASS |
| Chênh lệch weights giữa ranks mỗi sync | 0 | 0 |
| Payload mỗi rank mỗi sync | 5.387.156 bytes | 5.387.156 bytes |
| Max local relative L2 quantization error | 2,08835e-4 | 5,00747e-4 |
| Max averaged-delta relative L2 error vs FP32 | 2,04979e-4 | 4,08179e-4 |

Các maxima được lấy trên đủ 5 ranks × 3 sync. AllReduce vẫn dùng reduction
FP16, AllGather decode/weighted accumulate FP32. Sample counts mỗi sync là
`[10640, 240, 240, 240, 640]`, tổng `12000`; đúng batch `[266,6,6,6,16] × H=40`.
Mỗi rank ghi đúng 3 dòng telemetry với các trường số hữu hạn. Reference error
đo trên averaged delta, không trên weights sau cộng anchor.

Git lưu README này, `summary.json` và `summarize.py`. Các log, runtime/state
reports và CSV thô bên dưới chỉ lưu local; không nằm trong commit.

Bằng chứng local:

- `runtime.json`: phiên bản, layout model và optimizer thực tế.
- `transport.log`: kết quả regression 5 node.
- `allgather_state_training.log`, `allreduce_training.log`: training và kiểm tra state cuối.
- `allgather_state.json`, `allreduce_state.json`: optimizer iteration, dtype và model SHA256.
- `allgather/rankN/`, `allreduce/rankN/`: CSV lấy từ chính worker tương ứng.
- `summary.json`: số liệu tổng hợp đã kiểm tra bằng `summarize.py`.
- `allgather_training.log`: lượt AllGather ban đầu, trước khi thêm helper kiểm tra state cuối.

Chạy lại kiểm tra báo cáo khi còn đủ các raw artifacts local:

```bash
python HeteroViT-MPI/validation/delta_20261012/summarize.py
```

Các job chạy từ bản copy riêng `/tmp/heterovit_delta_validation_20261012` trên
mọi node, giữ nguyên worker repositories, process hiện có và MPI network flags.
Không gọi `pkill`, `rsync --delete` hay `run_5nodes.sh --sync`. Checkpoints mới
nằm trong output directories riêng dưới `HeteroViT-MPI/results/` trên lab01.
Thư mục staging trong `/tmp` cần dựng lại nếu node reboot hoặc bị cleanup.

Các lệnh dưới chạy từ workspace root và sử dụng bản staging đã kiểm chứng:

```bash
DELTA_MPI_ARGS=(
  -x PYTHONNOUSERSITE=1
  --mca btl_tcp_if_include 192.168.1.0/24
  --mca oob_tcp_if_include 192.168.1.0/24
  -np 5 --hostfile "$PWD/mpi_cluster/hostfile" --map-by ppr:1:node
  -wdir /tmp/heterovit_delta_validation_20261012
  "$PWD/mpi_cluster/python_by_host.sh"
)

mpiexec "${DELTA_MPI_ARGS[@]}" tools/preflight_delta_runtime.py configs/mpi/local_sgd_h40_fp16_delta_allgather_smoke.yaml
mpiexec "${DELTA_MPI_ARGS[@]}" -m unittest discover -s tests -p 'test_*comm*.py'

mpiexec "${DELTA_MPI_ARGS[@]}" tools/smoke_delta_training.py --config configs/mpi/local_sgd_h40_fp16_delta_allgather_smoke.yaml --max-steps 120 --seed 42
mpiexec "${DELTA_MPI_ARGS[@]}" tools/smoke_delta_training.py --config configs/mpi/local_sgd_h40_fp16_delta_allreduce_smoke.yaml --max-steps 120 --seed 42

# Bước tiếp theo: 40 epochs, reference checking tắt trong hai config này.
mpiexec "${DELTA_MPI_ARGS[@]}" train_mpi.py --config configs/mpi/local_sgd_h40_fp16_delta_allgather_40e.yaml --seed 42
mpiexec "${DELTA_MPI_ARGS[@]}" train_mpi.py --config configs/mpi/local_sgd_h40_fp16_delta_allreduce_40e.yaml --seed 42
```

40 epochs chưa được chạy trong lượt validation này. Khi so sánh với baseline,
sử dụng cùng cluster/environment/seed/H/batch/model/optimizer/LR; tắt reference
checking. Baseline whole-weight hiện ghi CSV hai lần mỗi sync với schema lệch;
cần lấy dòng hoàn chỉnh của trainer (hoặc `[WEIGHT SYNC]` trong run.log) khi
phân tích timing, không cộng trực tiếp toàn bộ dòng CSV. Logic baseline được
giữ nguyên. Smoke có reference traffic và hash/validation overhead; không dùng
các timing này để kết luận phương pháp nào nhanh hơn.
