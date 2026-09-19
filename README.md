# HeteroViT-MPI

Hệ thống huấn luyện phân tán đồng bộ (**Synchronous Distributed Training**) cho Vision Transformer (**ViT-Tiny**) trên cụm máy tính phần cứng không đồng nhất (**Heterogeneous 5-Node Cluster**) với MPI.

---

## 1. Cấu hình Cụm 5 Nodes (Cluster Topology)

| Rank | Hostname | IP | Thiết bị (Hardware) | Batch Size | Workload Share |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **0** | `iciplab01` | `192.168.1.131` | **2x NVIDIA Titan Z** (MirroredStrategy, Grad Accum $3 \times 128$) | **384** | 60.0% |
| **1** | `iciplab02` | `192.168.1.132` | Intel Pentium G3260 CPU (2C/2T, 8GB RAM) | **64** | 10.0% |
| **2** | `iciplab03` | `192.168.1.133` | Intel Core i3-560 CPU (2C/4T, 4GB RAM) | **48** | 7.5% |
| **3** | `iciplab04` | `192.168.1.134` | Intel Core i3-560 CPU (2C/4T, 8GB RAM) | **48** | 7.5% |
| **4** | `iciplab05` | `192.168.1.135` | Intel Core i5-7400 CPU (4C/4T, 8GB RAM) | **96** | 15.0% |

* **Global Batch Size**: $384 + 64 + 48 + 48 + 96 = \mathbf{640}$ mẫu/step.
* **Dataset**: CIFAR-10 ($45,000$ train, $5,000$ val) $\rightarrow$ Đúng **$70$ steps/epoch** trên toàn bộ 5 node.

---

## 2. Tính năng Cốt lõi (Key Features)

* **Static Uneven Workload Allocation**: Phân chia khối lượng dữ liệu và kích thước batch theo năng lực xử lý thực tế của từng node.
* **Gradient Accumulation trên GPU**: Phân tách batch 384 thành các micro-batch 128 để tối ưu hóa bộ nhớ, không gây tràn VRAM 6GB trên Titan Z.
* **Weighted Gradient AllReduce**: Đồng bộ gradient chính xác theo công thức trọng số mẫu:
  $$g_{\text{global}} = \frac{1}{640} \sum_{i=0}^{4} B_i \cdot g_i$$
* **Kiểm tra Đồng bộ Tự động (Sync Check)**: Kiểm tra độ lệch trọng số và gradient định kỳ mỗi 20 steps (ngưỡng dung sai $< 10^{-5}$).
* **Straggler & Timeline Profiling**: Đo lường và ghi log chi tiết thời gian tính toán (`compute_ms`), thời gian rảnh rỗi (`idle_wait_ms`) của từng rank ra file `ranks_timeline.csv`.
* **Fault Tolerance & Resume**: Tự động lưu checkpoint `best.weights.h5`, `last.weights.h5` và hỗ trợ `--resume` tiếp tục từ epoch gần nhất khi gặp sự cố.

---

## 3. Hướng dẫn Chạy (Quick Start)

Từ thư mục gốc dự án (`HeteroViT-Project`):

### A. Kiểm tra nhanh (Smoke Test 2 steps)
```bash
./mpi_cluster/run_5nodes.sh --sync --config HeteroViT-MPI/configs/mpi/hetero_static_5nodes.yaml --max-steps 2
```

### B. Huấn luyện Đầy đủ (Full Training 20 Epochs)
```bash
nohup ./mpi_cluster/run_5nodes.sh --sync --config HeteroViT-MPI/configs/mpi/hetero_static_5nodes.yaml > "logs/$(date +%Y%m%d_%H%M%S)_mpi_hetero_static_5nodes_full.log" 2>&1 & disown -h
```

### C. Tiếp tục huấn luyện từ Checkpoint dở dang (Resume)
```bash
nohup ./mpi_cluster/run_5nodes.sh --sync --config HeteroViT-MPI/configs/mpi/hetero_static_5nodes.yaml --resume > "logs/$(date +%Y%m%d_%H%M%S)_mpi_hetero_static_5nodes_resume.log" 2>&1 & disown -h
```

### D. Theo dõi tiến độ thời gian thực
```bash
tail -f $(ls -t logs/*.log | head -n 1)
```

---

## 4. Cấu trúc Thư mục

```text
HeteroViT-MPI/
├── train_mpi.py                       # Điểm vào chính của MPI distributed training
├── configs/mpi/
│   ├── baseline_5nodes.yaml           # Cấu hình baseline chia đều (128 mẫu/node)
│   └── hetero_static_5nodes.yaml      # Cấu hình chia tĩnh theo năng lực phần cứng
├── src/
│   ├── data/cifar10.py                # Data pipeline sharding theo tỷ lệ batch size
│   ├── models/vit.py                  # Mô hình Vision Transformer (ViT-Tiny) thuần TF/Keras
│   ├── training/trainer_mpi.py        # MPITrainer (Weighted AllReduce, Grad Accum, Timeline)
│   └── metrics/logger.py              # Quản lý kết quả, log và CSV metrics
└── results/                           # Thư mục lưu kết quả, timeline và checkpoints
```
