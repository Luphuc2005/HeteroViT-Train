# HeteroViT-MPI: Cluster MPI Configurations (`configs/mpi/`)

This directory contains the production YAML configuration files used for distributed training across the 5-node heterogeneous cluster:
- **Master (Rank 0)**: `iciplab01` (2x NVIDIA GeForce GTX TITAN Z GPU, batch = 266)
- **Workers (Rank 1-4)**: `iciplab02`..`iciplab05` (CPU nodes, batches = [6, 6, 6, 16])
- **Global Batch Size**: 300 (Effective total across 5 nodes)
- **Model**: ViT-Tiny (~2.69M parameters, 10.28 MB FP32 gradient payload)
- **Network**: 100 Mbps Fast Ethernet (~11.75 MB/s effective bandwidth)

---

## Active Production Configurations

| Config File | Synchronization Mode | Communication Primitive | Comm Frequency | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| [`auto_zero_idle_300.yaml`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/configs/mpi/auto_zero_idle_300.yaml) | Synchronous SGD (`gradient_allreduce`) | `MPI.Allreduce(SUM)` | Every step ($H=1$) | Clean Synchronous AllReduce Baseline |
| [`local_sgd_300.yaml`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/configs/mpi/local_sgd_300.yaml) | Local SGD (`local_sgd`) | `MPI.Allreduce(SUM)` | Every $H$ steps ($H \in \{4, 8, 16\}$) | Communication-efficient Local SGD |
| [`master_agg_300.yaml`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/configs/mpi/master_agg_300.yaml) | Master Aggregation (`master_aggregation`) | Point-to-point `Gather / Bcast` | Every step ($H=1$) | Parameter Server / Master Topology |

---

## Key Configuration Parameters

### 1. `auto_zero_idle_300.yaml` (AllReduce Baseline)
```yaml
training:
  global_batch_size: 300
  auto_balance: true          # Dynamic Zero-Idle load balancer (266, 6, 6, 6, 16)
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  drop_remainder: true

dynamic_scheduler:
  enabled: true               # Monitors node compute time to balance stragglers
  ema_alpha: 0.2
  slowdown_threshold_r: 1.15
```

### 2. `local_sgd_300.yaml` (Local SGD)
```yaml
training:
  global_batch_size: 300
  sync_mode: local_sgd        # Triggers Local SGD periodic weight averaging
  local_sgd_h: 4              # Synchronization period H (overridden via CLI --local-sgd-h)
  avg_policy: sample_weighted # Rank 0 weight = 266/300; Ranks 1-3 = 6/300; Rank 4 = 16/300
  optimizer_state_sync: preserve_local
```

---

## Archived Configurations (`configs/mpi/archive/`)
Legacy and diagnostic configurations from earlier development phases:
- `local_steps_k*_5nodes.yaml`: Early fixed-interval weight sync probes
- `val_c*_*.yaml`: Validation cluster scaling experiments
- `baseline_5nodes.yaml`, `hetero_static_5nodes.yaml`: Static batch configurations

