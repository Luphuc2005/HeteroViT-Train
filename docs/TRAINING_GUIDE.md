# Training Guide: HeteroViT Distributed & Local Benchmarks on ICIP Cluster

This document serves as the comprehensive user manual for training Vision Transformers (**ViT-Tiny**) on the ICIP server environment, encompassing both single-machine local benchmarks and heterogeneous 5-node distributed training via MPI.

---

## 1. Overview

**HeteroViT** is a distributed and heterogeneous training framework designed for deep neural networks (specifically Vision Transformers) operating over hardware with non-uniform compute capabilities and network constraints.

The project addresses system heterogeneity across two primary training topologies:
1. **Single-Node Execution (`train.py`)**: Runs purely on the master workstation (`iciplab01`), supporting:
   - CPU-only execution with fine-grained thread pinning (`CPUTrainer`).
   - Single-GPU execution (`GPUTrainer`).
   - Multi-GPU execution on dual NVIDIA Titan Z GPUs using `tf.distribute.MirroredStrategy` (`MultiGPUTrainer`).
   - Single-node Joint Hybrid execution where 2x Titan Z GPUs and 18 CPU cores compute concurrently and synchronize gradients via in-memory shared buffers (`JointTrainer`).
2. **Cluster Distributed Execution (`train_mpi.py`)**: Orchestrates distributed training across a 5-node cluster using Open MPI and `mpi4py`. It features:
   - **Static & Analytical Workload Balancing (Zero-Idle Balancer)**: Partitions global batch sizes (e.g., 300, 600) non-uniformly across ranks according to measured compute speeds to minimize idle synchronization barriers.
   - **Periodic Model Averaging (Local SGD)**: Computes local SGD steps across workers with periodic parameter averaging every $H$ steps, drastically reducing network transmission over 100 Mbps Ethernet.
   - **Adaptive Synchronization ($H$) Controllers**: Dynamically modulates synchronization interval $H$ via drift tracking (legacy `drift_adaptive_h`, Consensus-Budget Adaptive `cba_h`, Online Dynamics Drift `odd_h`, and Primal-Dual Consensus Adaptive `pdca_h`).
   - **Quantized Weight & Delta Communication**: Compresses synchronization payloads to FP16 full weights or FP16 weight deltas with optional Error Feedback (EF).

---

## 2. Requirements

The environment specifications below are extracted directly from the ICIP server cluster:

### Hardware Specifications

| Role | Hostname | IP Address | Compute Devices | CPU Model | Cores / Threads | System RAM | Network |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Master (Rank 0)** | `iciplab01` | `192.168.1.131` | **2x NVIDIA GeForce GTX TITAN Z** (6 GB VRAM each) | 2x Intel Xeon E5-2670 v3 | 24C / 48T | 80 GB | 100 Mbps / Gigabit |
| **Worker (Rank 1)** | `iciplab02` | `192.168.1.132` | CPU Only | Intel Pentium G3260 @ 3.30 GHz | 2C / 2T | 8 GB | 100 Mbps Ethernet |
| **Worker (Rank 2)** | `iciplab03` | `192.168.1.133` | CPU Only | Intel Core i3-560 @ 3.33 GHz | 2C / 4T | 4 GB | 100 Mbps Ethernet |
| **Worker (Rank 3)** | `iciplab04` | `192.168.1.134` | CPU Only | Intel Core i3-560 @ 3.33 GHz | 2C / 4T | 8 GB | 100 Mbps Ethernet |
| **Worker (Rank 4)** | `iciplab05` | `192.168.1.135` | CPU Only | Intel Core i5-7400 @ 3.00 GHz | 4C / 4T | 8 GB | 100 Mbps Ethernet |

### Operating System & Drivers (Master Node)
- **OS**: Ubuntu 20.04.6 LTS (Focal Fossa), Linux Kernel `5.4.0-216-generic` (x86_64)
- **NVIDIA GPU Driver**: `470.256.02`
- **CUDA Toolkit**: CUDA 11.4 (`/usr/local/cuda-11.4/bin/nvcc`)
- **MPI Runtime**: Open MPI 4.0.3 (`/usr/local/openmpi-4.0.3/bin/mpirun`)

### Python & Key Libraries

| Dependency | Required Version in Repo | Master Environment (`iciplab01`) | Worker Nodes (`iciplab02..05`) |
| :--- | :--- | :--- | :--- |
| **Python** | 3.8.x | `3.8.18` (`/home/icip/Ha/myenv`) | `3.8.x` (`/home/icip/miniconda3/envs/heterovit_mpi`) |
| **TensorFlow** | `>=2.15.0` (Repo spec) / `2.10.1` (Actual) | `2.10.1` (`keras 2.10.0`) | `2.10.1` / `2.12.0` |
| **mpi4py** | `>=4.0` | `4.1.1` | `4.1.1` |
| **NumPy** | `>=1.23.5, <2.0` | `1.23.5` | `1.22.x` - `1.24.x` |
| **PyYAML** | `>=6.0` | `6.0.3` | Installed |
| **pandas** | `>=2.0.0` | `2.0.3` | Installed |
| **psutil** | `>=5.9.0` | `7.2.2` | Installed |
| **matplotlib** | `>=3.7.0` | `3.7.5` | Installed |

> [!IMPORTANT]
> **Titan Z GPU Architecture Notice**: NVIDIA GeForce GTX TITAN Z uses the **Kepler GK110B** architecture (Compute Capability 3.5). The highest CUDA driver supported is 470.xx, and the maximum compatible TensorFlow version with full GPU acceleration on this architecture is **TensorFlow 2.10.1**.

---

## 3. Environment Setup

### 3.1 Activating Existing Environment on ICIP Master (`iciplab01`)

The master node already has a pre-configured Python virtual environment with CUDA 11.4, TensorFlow 2.10.1, and Open MPI 4.0.3 bindings.

Execute:
```bash
# 1. Activate the pre-installed virtual environment
source /home/icip/Ha/myenv/bin/activate

# 2. CRITICAL: Isolate environment from ~/.local site-packages
# Failure to set this will cause ModuleNotFoundError: No module named 'six'
export PYTHONNOUSERSITE=1
```

### 3.2 Setting Up a Fresh Virtual Environment (Optional / Reproducibility)

If you need to reconstruct the virtual environment from scratch:

```bash
# Create python 3.8 virtual environment
python3.8 -m venv ~/my_heterovit_env
source ~/my_heterovit_env/bin/activate
export PYTHONNOUSERSITE=1

# Install wheel & core dependencies
pip install --upgrade pip setuptools wheel
pip install tensorflow==2.10.1
pip install numpy==1.23.5 pyyaml==6.0.3 psutil pandas matplotlib

# Install mpi4py linked against system Open MPI 4.0.3
MPICC=/usr/local/openmpi-4.0.3/bin/mpicc pip install --no-cache-dir --no-binary=mpi4py mpi4py
```

### 3.3 Verifying CUDA, GPU, and MPI

Run the following diagnostic commands on `iciplab01`:

```bash
# 1. Check NVIDIA driver and GPUs
nvidia-smi

# Expected output:
# Two NVIDIA GeForce GTX TITAN Z GPUs, each ~6083 MiB VRAM, Driver 470.256.02, CUDA 11.4

# 2. Verify CUDA compiler
/usr/local/cuda-11.4/bin/nvcc --version

# 3. Verify TensorFlow GPU detection
PYTHONNOUSERSITE=1 python -c "import tensorflow as tf; print('TF Version:', tf.__version__); print('GPUs detected:', tf.config.list_physical_devices('GPU'))"

# Expected output:
# TF Version: 2.10.1
# GPUs detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU')]

# 4. Verify Open MPI and mpi4py
PYTHONNOUSERSITE=1 python -c "from mpi4py import MPI; print('MPI Vendor:', MPI.get_vendor(), '| Comm Size:', MPI.COMM_WORLD.Get_size())"

# Expected output:
# MPI Vendor: ('Open MPI', (4, 0, 3)) | Comm Size: 1
```

---

## 4. Dataset Preparation

### 4.1 Dataset Location & Formats

The CIFAR-10 data loader is implemented in [`HeteroViT-MPI/src/data/cifar10.py`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/src/data/cifar10.py). It automatically loads and partitions the dataset.

The loader supports two formats:
1. **Official Python Batch Directory**: Contains files `data_batch_1`, `data_batch_2`, `data_batch_3`, `data_batch_4`, `data_batch_5`, and `test_batch`.
2. **NumPy Archive**: A single `cifar10.npz` containing arrays `x_train`, `y_train`, `x_test`, `y_test`.

On the ICIP server, the dataset is already downloaded and unpacked at:
`/home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/data/cifar10/`

### 4.2 Directory Tree

```text
/home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/data/cifar10/
└── cifar-10-batches-py/
    ├── batches.meta
    ├── data_batch_1      # 10,000 training images
    ├── data_batch_2      # 10,000 training images
    ├── data_batch_3      # 10,000 training images
    ├── data_batch_4      # 10,000 training images
    ├── data_batch_5      # 10,000 training images
    ├── readme.html
    └── test_batch        # 10,000 evaluation images
```

### 4.3 Configuring Dataset Path in Config YAML

In any training configuration file, specify the dataset path under `dataset:`:

```yaml
dataset:
  name: cifar10
  data_dir: /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/data/cifar10  # Absolute or relative path
  num_classes: 10
```

When running distributed MPI training, ensure the dataset exists on all slave nodes at the identical path, or allow `./mpi_cluster/run_5nodes.sh --sync` to synchronize the directory.

---

## 5. Basic Training

### 5.1 Simplest Single-Node Runnable Command

To run a fast, single-node CPU baseline verification test on the master workstation:

```bash
cd /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

python train.py --config configs/cpu/cpu_06cores_20e.yaml
```

**Parameter Explanation**:
- `train.py`: Single-node training entry point.
- `--config configs/cpu/cpu_06cores_20e.yaml`: Specifies the YAML configuration file that defines model architecture (ViT-Tiny), CPU thread count (6 threads), batch size (128), and epoch count (20).

### 5.2 Simplest Distributed Cluster Smoke Test

To verify connectivity across all 5 nodes and run a 2-step distributed training test:

```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./mpi_cluster/run_5nodes.sh --sync --config HeteroViT-MPI/configs/mpi/auto_zero_idle_300.yaml --max-steps 2
```

**Parameter Explanation**:
- `./mpi_cluster/run_5nodes.sh`: Cluster launcher script that wraps `mpirun` across 5 nodes.
- `--sync`: Automatically kills zombie MPI processes, syncs latest code/hostfile across all nodes via `rsync`/`scp`.
- `--config HeteroViT-MPI/configs/mpi/auto_zero_idle_300.yaml`: Baseline distributed configuration with global batch size 300.
- `--max-steps 2`: Stops after 2 steps (smoke test).

---

## 6. Training Configurations

The project supports specific training paradigms, while others are intentionally not implemented due to hardware constraints.

| Training Mode | Supported in Codebase? | Entry Point & Implementation | Details |
| :--- | :--- | :--- | :--- |
| **FP32 Training** | **Supported** | `train.py`, `train_mpi.py` | Full 32-bit floating point model computation and default weight communication (`comm_precision: fp32`). |
| **FP16 Communication** | **Supported** | `train_mpi.py` (`src/training/fp16_communicator.py`, `fp16_delta_communicator.py`) | Compresses inter-node weight exchange to IEEE 754 float16 during Local SGD synchronization (`comm_precision: fp16`, `fp16_delta_allgather`, `fp16_delta_allreduce`). |
| **BF16 (bfloat16)** | **Currently not supported** | N/A | Hardware constraint: Titan Z (Kepler GK110B) lacks BF16 instruction support. |
| **Mixed Precision Computation (AMP)** | **Currently not supported** | N/A | Local forward/backward compute is kept in FP32; `tf.keras.mixed_precision` is not used. |
| **Single GPU** | **Supported** | `train.py` (`src/training/trainer_gpu.py: GPUTrainer`) | Runs on `/GPU:0` via `mode: gpu` in config. |
| **Multi-GPU (Mirrored)** | **Supported** | `train.py` (`src/training/trainer_gpu.py: MultiGPUTrainer`) | Distributes training across both Titan Z GPUs via `tf.distribute.MirroredStrategy`. |
| **Single-Node Joint Hybrid (CPU+GPU)** | **Supported** | `train.py` (`src/training/trainer_joint.py: JointTrainer`) | Asynchronously merges GPU worker (MirroredStrategy) with dedicated CPU worker via OS shared memory. |
| **Distributed Synchronous SGD** | **Supported** | `train_mpi.py` (`src/training/trainer_mpi.py`) | Every-step weighted gradient AllReduce (`sync_mode: gradient_allreduce`). |
| **Distributed Local SGD** | **Supported** | `train_mpi.py` (`src/training/trainer_mpi.py`) | Periodic model parameter averaging every $H$ steps with sample-weighted consensus (`sync_mode: local_sgd`). |
| **Adaptive Local SGD** | **Supported** | `train_mpi.py` (`src/training/adaptive/`) | Dynamic $H$ adjustment: V1 Consensus Budget (`cba_h`), V2 Online Dynamics (`odd_h`), V3 Primal-Dual (`pdca_h`). |

---

## 7. Local SGD & Distributed Training

### 7.1 Concepts & Architecture

- **World Size**: 5 nodes (ranks 0 to 4).
- **Master Node (Rank 0 - `iciplab01`)**: Equipped with 2x NVIDIA Titan Z GPUs. In single-rank multi-GPU mode, rank 0 processes the majority of the global batch (e.g., 266 out of 300 samples, split $133 \times 2$ across GPUs).
- **Worker Nodes (Ranks 1 to 4 - `iciplab02..05`)**: Pure CPU worker nodes processing smaller mini-batches (e.g., 6, 6, 6, 16 samples) to match execution latencies.
- **Local SGD Synchronization Interval ($H$)**: Instead of communicating gradients every step, each node executes $H$ steps of standard local SGD using its local data partition. At every $H$-th step, ranks synchronize model weights:
  $$\mathbf{w}_{\text{global}} = \sum_{r=0}^{4} \frac{B_r}{B_{\text{global}}} \mathbf{w}_r$$
- **Averaging Policy (`--avg-policy`)**:
  - `sample_weighted` (default): Weights each rank's contribution by its local batch size ($B_r / B_{\text{global}}$). Essential for heterogeneous clusters.
  - `uniform`: Arithmetic mean ($1/N$).
- **Communication Backends (`--comm-backend`)**:
  - `allreduce` (default): Ring AllReduce collective over Open MPI (`MPI.COMM_WORLD.Allreduce`).
  - `master_aggregation` (or `ps`): Point-to-point Gather to rank 0, multi-threaded aggregation, and Broadcast back.

### 7.2 Launching Distributed Training

Distributed jobs are initiated from `iciplab01` using `mpi_cluster/run_5nodes.sh`.

#### Interactive Execution
```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./mpi_cluster/run_5nodes.sh \
  --sync \
  --config HeteroViT-MPI/configs/mpi/local_sgd_h40_300_40e.yaml \
  --epochs 40 \
  --local-sgd-h 40 \
  --avg-policy sample_weighted \
  --comm-precision fp32
```

#### Production Background Execution (`nohup`)
```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

nohup ./mpi_cluster/run_5nodes.sh \
  --sync \
  --config HeteroViT-MPI/configs/mpi/local_sgd_h40_300_40e.yaml \
  --epochs 40 \
  > "HeteroViT-MPI/logs/$(date +%Y%m%d_%H%M%S)_local_sgd_h40_40e.log" 2>&1 & disown -h
```

---

## 8. Important Training Arguments

### 8.1 Single-Node Training (`train.py`)

| Argument | Meaning | Default | Example |
| :--- | :--- | :--- | :--- |
| `--config` | Path to YAML configuration file *(required)* | N/A | `--config configs/gpu/gpu_2titanz_b256_20e.yaml` |

*(All hyper-parameters for `train.py` are set directly inside the config YAML file).*

---

### 8.2 Distributed MPI Training (`train_mpi.py` & `run_5nodes.sh`)

The table below lists all CLI flags supported by `train_mpi.py`:

| Argument | Meaning | Default in Code | Example Value |
| :--- | :--- | :--- | :--- |
| `--config` | Path to YAML configuration file *(required)* | `baseline_5nodes.yaml` | `--config configs/mpi/local_sgd_h40_300_40e.yaml` |
| `--epochs` | Total training epochs (overrides config) | `None` (uses config) | `--epochs 40` |
| `--max-steps` | Stop training after $N$ steps (smoke tests) | `None` (full training) | `--max-steps 10` |
| `--resume` | Resume checkpoint (`auto` or file/dir path) | `None` | `--resume` or `--resume results/run_xxx/` |
| `--start-epoch` | Starting epoch when resuming | Auto from `train.csv` | `--start-epoch 21` |
| `--name` | Experiment name override for output dir | `None` (uses config) | `--name local_sgd_h40_run2` |
| `--seed` | Random seed for reproducibility | `42` | `--seed 43` |
| `--global-batch` | Target global batch size across cluster | `300` | `--global-batch 300` |
| `--auto-balance` | Enable Zero-Idle dynamic analytical balancing | `False` | `--auto-balance` |
| `--sync-mode` | Synchronization paradigm | `gradient_allreduce` | `--sync-mode local_sgd` |
| `--local-sgd-h` | Local SGD synchronization interval $H$ | `1` | `--local-sgd-h 40` |
| `--avg-policy` | Parameter averaging policy | `sample_weighted` | `--avg-policy sample_weighted` |
| `--local-steps` | Local steps $K$ before delta sync | `10` | `--local-steps 20` |
| `--opt-sync` | Optimizer state synchronization mode | `preserve_local` | `--opt-sync preserve_local` |
| `--comm-backend` | Gradient/model communication primitive | `allreduce` | `--comm-backend allreduce` |
| `--ps-agg-threads` | Aggregation threads on rank 0 for master backend | `None` (12) | `--ps-agg-threads 12` |
| `--comm-precision` | Precision of model sync payloads | `fp32` | `--comm-precision fp16` |
| `--comm-collective`| Collective algorithm for FP16 sync | `allreduce` | `--comm-collective allgather` |
| `--local-sgd-policy`| Local SGD policy controller | `fixed` | `--local-sgd-policy pdca_h` |
| `--adaptive-type` | Adaptive controller formulation | `drift_adaptive_h` | `--adaptive-type pdca_h` |
| `--adaptive-candidates` | Candidate $H$ values for adaptive controller | `None` | `--adaptive-candidates 40,80,120,150` |
| `--adaptive-initial-h` | Starting interval $H$ for adaptive mode | `40` | `--adaptive-initial-h 40` |
| `--adaptive-budget` | Consensus risk budget ($B$ or $\Gamma$) | `None` | `--adaptive-budget 0.05` |
| `--adaptive-dual-lr` | Dual step size $\eta_\mu$ for V3 controller | `None` (1.0) | `--adaptive-dual-lr 1.0` |
| `--dynamic-rebalance`| Enable Phase 2 closed-loop load rebalancing | `False` | `--dynamic-rebalance` |

---

## 9. Output

Every training run automatically generates a self-contained, timestamped experiment directory (using Vietnam Time `UTC+7`, e.g., `results/<experiment_name>_YYYYMMDD_HHMMSS/`).

### Directory Layout

```text
results/local_sgd_h40_300_40e_20261019_131721/
├── checkpoints/
│   ├── best.weights.h5             # Checkpoint with highest Top-1 Validation Accuracy
│   └── last.weights.h5             # Most recent epoch checkpoint (for resumption)
├── config.yaml                     # Snapshot of active YAML configuration
├── metadata.json                   # System hardware, CPU affinity, and library versions
├── run.log                         # Complete text log output
├── train.csv                       # Epoch-level metrics table
├── ranks_timeline.csv              # Per-rank compute, idle wait, and sync latencies (MPI only)
├── round_diagnostic_rank0.csv      # Round-level Local SGD weight drift diagnostics
└── step_timing_breakdown.csv       # Microsecond step timers
```

### Logged Metrics in `train.csv`
- `epoch`: Current epoch index ($1 \dots N$).
- `train_loss`: Cross-entropy loss on training set.
- `train_accuracy`: Top-1 training accuracy.
- `val_loss`: Cross-entropy loss on 5,000 validation images.
- `val_accuracy`: Top-1 validation accuracy.
- `epoch_time`: Total wall-clock time for the epoch in seconds.
- `train_time`: Wall-clock time spent inside the training step loop.
- `val_time`: Wall-clock time spent evaluating validation split.
- `samples_per_sec`: Effective throughput (samples/sec).
- `learning_rate`: Active learning rate.

### TensorBoard & WandB Support
- **TensorBoard**: **Currently not supported** (metrics are recorded to `train.csv` and `ranks_timeline.csv`).
- **Weights & Biases (WandB)**: **Currently not supported**.

---

## 10. Resume Training

Resuming from saved checkpoints is supported for distributed training (`train_mpi.py`).

### 10.1 Automatic Resume (Latest Run)
If an experiment is interrupted, use `--resume` without arguments. The launcher searches `results/` for the latest folder matching the experiment prefix, recovers `checkpoints/last.weights.h5`, and reads `train.csv` to resume from the next epoch:

```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./mpi_cluster/run_5nodes.sh \
  --sync \
  --config HeteroViT-MPI/configs/mpi/local_sgd_h40_300_40e.yaml \
  --resume
```

### 10.2 Resuming from an Explicit Checkpoint Path

```bash
./mpi_cluster/run_5nodes.sh \
  --sync \
  --config HeteroViT-MPI/configs/mpi/local_sgd_h40_300_40e.yaml \
  --resume HeteroViT-MPI/results/local_sgd_h40_300_40e_20261019_131721/checkpoints/last.weights.h5 \
  --start-epoch 21
```

> [!WARNING]
> **Error Feedback Resume Restriction**:
> If `--comm-precision fp16_delta_allgather` is used with Error Feedback (`comm_error_feedback: true`), `--resume` is explicitly disallowed by the trainer (`ValueError`). EF experiments maintain per-rank memory residuals that cannot be recovered from weight files alone and must be run fresh.
> Single-node `train.py` currently does not support resuming via CLI.

---

## 11. Monitoring Training

### 11.1 GPU and System Telemetry

```bash
# Monitor master node GPUs every 1 second
watch -n 1 nvidia-smi

# Check CPU core utilization and thread loads
htop
```

### 11.2 Real-time Training Logs

```bash
# Stream the most recent training log
tail -f $(ls -t /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/logs/*.log 2>/dev/null | head -n 1)

# Or tail the active run's run.log inside results
tail -f $(ls -td /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/*/ 2>/dev/null | head -n 1)run.log
```

### 11.3 Project-Specific Live Monitor

For single-node Joint training, the project provides an interactive live dashboard:

```bash
cd /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI
python tools/monitor_joint.py
```

---

## 12. Example Experiments

### Experiment 1 — Single-Node Dual Titan Z GPU Baseline (20 Epochs)
Runs multi-GPU training on `iciplab01` using `tf.distribute.MirroredStrategy` without MPI:

```bash
cd /home/icip/Ha/HeteroViT-Project/HeteroViT-MPI
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0,1

python train.py --config configs/gpu/gpu_2titanz_b256_20e.yaml
```

### Experiment 2 — Clean Synchronous AllReduce Baseline (5 Nodes, 20 Epochs)
Executes standard distributed synchronous SGD with step-by-step AllReduce:

```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./HeteroViT-MPI/scripts/run_01_allreduce_20e.sh 20
```

### Experiment 3 — Standard Local SGD ($H = 40$, FP32, 40 Epochs)
The primary production baseline: periodic model averaging every 40 steps with sample weighting:

```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./HeteroViT-MPI/scripts/run_11_local_sgd_h40_40e.sh 40
```

### Experiment 4 — Quantized FP16 Local SGD ($H = 40$, FP16 AllReduce, 40 Epochs)
Compresses model parameter synchronization to float16, halving Ethernet communication volume:

```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./HeteroViT-MPI/scripts/run_21_local_sgd_h40_fp16_40e.sh 40
```

### Experiment 5 — Adaptive Local SGD (PDCA V3 Controller, 40 Epochs)
Dynamically selects $H \in \{40, 80, 120, 150\}$ using the Primal-Dual Consensus Adaptive algorithm:

```bash
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

./HeteroViT-MPI/scripts/run_20_local_sgd_adaptive_v3_pdca_40e.sh 40
```

---

## 13. Troubleshooting

### 1. `ModuleNotFoundError: No module named 'six'`
- **Cause**: Python is attempting to load user packages from `~/.local` that conflict with the virtualenv.
- **Fix**: Always export `PYTHONNOUSERSITE=1` before invoking Python:
  ```bash
  export PYTHONNOUSERSITE=1
  ```

### 2. CUDA Out of Memory (OOM) on Titan Z
- **Cause**: Titan Z has 6 GB VRAM per GPU die. Allocating large batches (>128 per GPU) causes allocation failure.
- **Fix**:
  1. Ensure `memory_growth: true` is set in the config YAML.
  2. For large rank 0 batches (e.g. 266 or 384), activate gradient accumulation in config:
     ```yaml
     training:
       grad_accum:
         enabled: true
         micro_batch_size: 128
     ```

### 3. NCCL Errors / Misunderstandings
- **Note**: HeteroViT does **not** use NVIDIA NCCL for inter-node communication. Inter-node coordination is handled via **Open MPI over TCP (`mpi4py`)**.
- **Fix**: If an Open MPI error occurs, check Ethernet subnet MCA parameters:
  `--mca btl_tcp_if_include 192.168.1.0/24 --mca oob_tcp_if_include 192.168.1.0/24`

### 4. Distributed Training Hangs or Zombie Processes
- **Cause**: A previous crashed MPI run left orphan `orted` or `python` processes holding ports or sockets on slave nodes.
- **Fix**: Run the emergency cluster termination script:
  ```bash
  cd /home/icip/Ha/HeteroViT-Project
  ./mpi_cluster/kill_cluster.sh
  ```
  Or re-run with the `--sync` flag in `run_5nodes.sh`, which automatically cleans up slave processes.

### 5. Checkpoint Loading Inconsistency (`ValueError`)
- **Cause**: Invoking `--resume` on an Error Feedback run (`comm_error_feedback: true`).
- **Fix**: Error Feedback experiments must start from epoch 1 without `--resume`.

### 6. Dataset Not Found (`FileNotFoundError`)
- **Cause**: Path in config YAML does not match physical path on the node.
- **Fix**: Verify that `/home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/data/cifar10` exists and contains `cifar-10-batches-py/` or `cifar10.npz`.

---

## 14. Quick Start

For users with the ICIP environment already configured, copy and paste these commands to launch training immediately:

```bash
# 1. Navigate to project root & activate environment
cd /home/icip/Ha/HeteroViT-Project
source /home/icip/Ha/myenv/bin/activate
export PYTHONNOUSERSITE=1

# 2. Run 5-Node Local SGD Distributed Training (H=40, 40 Epochs) in background
nohup ./HeteroViT-MPI/scripts/run_11_local_sgd_h40_40e.sh 40 > train.log 2>&1 & disown -h

# 3. Stream real-time training progress
tail -f train.log
```
