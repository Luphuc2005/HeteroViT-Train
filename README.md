# Heterogeneous Distributed Training for Vision Transformer (ViT)

Dự án nghiên cứu thực nghiệm tối ưu hóa huấn luyện phân tán không đồng nhất (Heterogeneous Distributed Training) cho Vision Transformer (ViT) trên tập dữ liệu **CIFAR-10**, sử dụng **TensorFlow/Keras** thuần và kiến trúc custom training loop.

---

## 1. Lộ trình thực nghiệm (Research Roadmap)

Thực nghiệm được thiết kế thành các pha lũy tiến có hệ thống:

```text
[GIAI ĐOẠN HIỆN TẠI]
1. CPU-only trên 1 GPU node (6, 12, 18, 24 physical cores)
   ↓
2. 1 GPU V100-only (Baseline tham chiếu)

[CÁC GIAI ĐOẠN TIẾP THEO]
   ↓
3. CPU + 1 V100 (Heterogeneous hybrid)
   ↓
4. CPU + 1 V100 Adaptive Workload Scheduling (Cân bằng tải theo throughput)
   ↓
5. CPU + 1 V100 Loss-Aware Scheduling (Phân phối tải theo chất lượng gradient/loss)
   ↓
6. 2 V100 Multi-Node (Phân tán đồng nhất nhiều node)
   ↓
7. CPU + 2 V100 Multi-Node (Phân tán không đồng nhất trên cụm)
```

> **Lưu ý**: Giai đoạn này tập trung chuẩn hóa môi trường, benchmark độc lập **CPU-only** và **1 GPU V100** trên cùng 1 GPU node vật lý để bảo đảm tính nhất quán của phần cứng.

---

## 2. Cấu trúc thư mục dự án

```text
hetero_vit_training/
├── train.py                  # Entry point chính tiếp nhận config và điều phối trainer
├── requirements.txt          # Danh sách thư viện phụ thuộc
├── README.md                 # Tài liệu hướng dẫn chi tiết
│
├── configs/                  # Chứa toàn bộ cấu hình YAML thực nghiệm
│   ├── cpu/
│   │   ├── cpu_06cores.yaml  # Config cho 6 CPU cores
│   │   ├── cpu_12cores.yaml  # Config cho 12 CPU cores
│   │   ├── cpu_18cores.yaml  # Config cho 18 CPU cores
│   │   └── cpu_24cores.yaml  # Config cho 24 CPU cores
│   └── gpu/
│       └── gpu_1v100.yaml    # Config cho 1 GPU NVIDIA Tesla V100
│
├── src/                      # Toàn bộ mã nguồn cốt lõi
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar10.py        # Data pipeline độc lập, offline check, tf.data optimize
│   │   └── augmentation.py   # Data augmentation chuẩn cho CIFAR-10
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── vit.py            # ViT-Tiny thuần TensorFlow/Keras (Patch + Transformer Blocks)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── base_trainer.py   # Base trainer, GradientTape custom loop, train_step tách biệt
│   │   ├── trainer_cpu.py    # CPU-specific trainer (Thread affinity & CPU devices)
│   │   └── trainer_gpu.py    # GPU-specific trainer (Memory growth & V100 context)
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── logger.py         # Quản lý run directory, file log, CSV logger
│   │   └── system_metrics.py # Thu thập hardware metadata (CPU affinity, GPU model, v.v.)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # YAML parser và ConfigDict
│       └── seed.py           # Thiết lập seed cố định cho Python, NumPy, TensorFlow
│
├── scripts/                  # Shell scripts chạy thực nghiệm với taskset & env vars
│   ├── run_cpu_06.sh         # Run script 6 cores
│   ├── run_cpu_12.sh         # Run script 12 cores
│   ├── run_cpu_18.sh         # Run script 18 cores
│   ├── run_cpu_24.sh         # Run script 24 cores
│   └── run_gpu_1v100.sh      # Run script 1 V100 GPU
│
├── experiments/              # Ghi chép và tài liệu theo từng pha thực nghiệm
│   ├── 01_cpu_only/
│   └── 02_gpu_only/
│
└── results/                  # Kết quả thực nghiệm (không ghi đè, định dạng timestamp)
    ├── logs/
    ├── csv/
    ├── checkpoints/
    └── figures/
```

---

## 3. Nơi đặt Dataset CIFAR-10

Hệ thống **KHÔNG tự động tải CIFAR-10** từ Internet nhằm đảm bảo tuân thủ môi trường tính toán offline/cluster.

Vị trí mặc định trong config:
```text
hetero_vit_training/data/cifar10/
```

Bạn có thể chuẩn bị CIFAR-10 theo 1 trong 2 cách sau:

### Cách 1: Thư mục Python Batches chính thức (Khuyến nghị)
Tải `cifar-10-python.tar.gz` từ trang của Alex Krizhevsky và giải nén vào thư mục `data/cifar10/`. Cấu trúc bên trong:
```text
data/cifar10/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```
*(Hỗ trợ cả trường hợp thư mục giải nén tên là `data/cifar10/cifar-10-batches-py/`)*.

### Cách 2: File NumPy nén (`cifar10.npz`)
Lưu file `cifar10.npz` chứa 4 arrays: `x_train, y_train, x_test, y_test` vào:
```text
data/cifar10/cifar10.npz
```

Nếu chưa có dataset tại vị trí đã chỉ định, chương trình sẽ dừng và thông báo rõ:
```text
CIFAR-10 dataset not found at <path>.
Please prepare the dataset before training.
```

---

## 4. Kiến trúc ViT-Tiny (CIFAR-10)

Mô hình được viết thuần bằng `tf.keras.layers`:
- **Input shape**: `32 x 32 x 3`
- **Patch size**: `4 x 4` (Tổng số patches: \((32/4)^2 = 64\))
- **Embedding Dimension**: `192`
- **CLS Token**: Learnable parameter shape `(1, 1, 192)`
- **Positional Embedding**: Learnable parameter shape `(1, 65, 192)`
- **Transformer Encoder**: 6 khối (`depth: 6`)
  - Multi-Head Attention: 3 heads (`key_dim: 64`)
  - Feed-Forward MLP: Dimension `768`, hàm kích hoạt `GELU`
  - Pre-LayerNorm và Residual connections
- **Head**: LayerNorm \(\rightarrow\) CLS representation \(\rightarrow\) Dense(`10 classes`, logits).
- **Tổng tham số**: ~2.8 triệu tham số (phù hợp năng lực tính toán của CPU và V100).

---

## 5. Hướng dẫn chạy thực nghiệm

Tất cả CPU experiment và GPU experiment đều chạy trên cùng **một GPU node** (tối đa 24 CPU cores vật lý). CPU affinity được điều khiển qua `taskset` trong shell script. Khi chạy CPU-only, GPU bị vô hiệu hóa hoàn toàn bằng `CUDA_VISIBLE_DEVICES=""`.

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### 1. CPU-Only Baseline (6 Cores)
```bash
bash scripts/run_cpu_06.sh
```
*Lệnh tương đương:*
```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=6 \
MKL_NUM_THREADS=6 \
taskset -c 0-5 \
python train.py --config configs/cpu/cpu_06cores.yaml
```

### 2. CPU-Only Baseline (12 Cores)
```bash
bash scripts/run_cpu_12.sh
```
*Lệnh tương đương:*
```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=12 \
MKL_NUM_THREADS=12 \
taskset -c 0-11 \
python train.py --config configs/cpu/cpu_12cores.yaml
```

### 3. CPU-Only Baseline (18 Cores)
```bash
bash scripts/run_cpu_18.sh
```
*Lệnh tương đương:*
```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=18 \
MKL_NUM_THREADS=18 \
taskset -c 0-17 \
python train.py --config configs/cpu/cpu_18cores.yaml
```

### 4. CPU-Only Baseline (24 Cores)
```bash
bash scripts/run_cpu_24.sh
```
*Lệnh tương đương:*
```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=24 \
MKL_NUM_THREADS=24 \
taskset -c 0-23 \
python train.py --config configs/cpu/cpu_24cores.yaml
```

### 5. GPU Baseline (1 NVIDIA V100)
```bash
bash scripts/run_gpu_1v100.sh
```
*Lệnh tương đương:*
```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py --config configs/gpu/gpu_1v100.yaml
```

---

## 6. Kết quả và Metrics ghi nhận

Mỗi lần chạy sẽ tạo một thư mục độc lập theo timestamp trong `results/`, ví dụ:
```text
results/
└── cpu_12cores_20260914_163000/
    ├── config.yaml          # Bản sao cấu hình YAML của lần chạy
    ├── metadata.json        # Metadata phần cứng (hostname, CPU affinity, GPU, TF version)
    ├── run.log              # Log chi tiết quá trình huấn luyện
    ├── train.csv            # File CSV ghi lại metrics qua các epoch
    └── checkpoints/
        ├── best.weights.h5  # Weights có val_accuracy cao nhất
        └── last.weights.h5  # Weights của epoch cuối cùng
```

### Định dạng `train.csv`:
```csv
epoch,train_loss,train_accuracy,val_loss,val_accuracy,epoch_time,samples_per_sec
```

---

## 7. Thiết kế mở rộng cho Hybrid Heterogeneous Training

Hệ thống đã tách biệt hoàn toàn hàm tính gradient:
```python
def train_step(model, optimizer, images, labels, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, accuracy, gradients
```

Trong các giai đoạn tiếp theo, kiến trúc mở rộng sẽ bổ sung:
- `src/training/trainer_hybrid.py`: Điều phối luồng làm việc giữa CPU worker và GPU worker.
- `src/scheduler/`:
  - `static_scheduler.py`: Phân chia batch cố định theo tỷ lệ (ví dụ 20% CPU - 80% GPU).
  - `throughput_scheduler.py`: Thích ứng tỷ lệ batch động dựa trên tốc độ xử lý thực tế (`samples_per_sec`).
  - `loss_aware_scheduler.py`: Điều phối tải dựa trên loss gradient divergence giữa các worker.
- `Global aggregation`: Tổng hợp local gradients từ CPU và GPU trước khi cập nhật trọng số toàn cục.
