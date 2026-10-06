# HeteroViT Experiment Results (`results/`)

This directory stores experimental results, logs, metrics, and checkpoints for HeteroViT distributed cluster experiments.

---

## 1. Active Benchmark Suite (20 Epochs, Batch 300)

The main benchmark evaluates ViT-Tiny across the 5-node heterogeneous cluster (`iciplab01` GPU master + `iciplab02`..`05` CPU workers) under 100 Mbps Fast Ethernet:

| Directory Pattern | Method | Comm Payload / Round | Sync Freq | Target Metric |
| :--- | :--- | :--- | :--- | :--- |
| `allreduce_baseline_300_20e_*` | Synchronous Ring AllReduce | 10.28 MB (FP32) | Every step ($H=1$) | Clean Synchronous Baseline |
| `local_sgd_h04_300_20e_*` | Local SGD ($H=4$) | 10.28 MB (FP32) | Every 4 steps | 4x Comm reduction |
| `local_sgd_h08_300_20e_*` | Local SGD ($H=8$) | 10.28 MB (FP32) | Every 8 steps | 8x Comm reduction |
| `local_sgd_h16_300_20e_*` | Local SGD ($H=16$) | 10.28 MB (FP32) | Every 16 steps | 16x Comm reduction |

Each run directory contains:
- `config.yaml`: Frozen configuration for reproducibility.
- `train.csv`: Step-by-step and epoch-by-epoch loss, top-1 accuracy, throughput, and step time.
- `run.log`: Full console output including per-step rank timing breakdown table.
- `cluster_resources.csv`: Per-epoch hardware metrics (CPU%, RAM, GPU VRAM) across all 5 nodes.
- `checkpoints/`: Model weights (`best.weights.h5`, `last.weights.h5`).

To aggregate metrics into a formatted comparison table:
```bash
python3 scripts/aggregate_benchmark_results.py
```

---

## 2. Archive Directory (`results/archive/`)

Historical and exploratory experimental data are categorized into:

- [`audit_verification_1epoch/`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/archive/audit_verification_1epoch/): Verified 1-epoch runs proving zero 6.5s overhead (`clean_allreduce_300`, `clean_local_sgd_h01_300`).
- [`phase2_dynamic_rebalance/`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/archive/phase2_dynamic_rebalance/): Dynamic scheduler & straggler rebalancer evaluations.
- [`phase1_local_steps/`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/archive/phase1_local_steps/): Initial weight averaging probes on 5 nodes.
- [`phase0_single_node/`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/archive/phase0_single_node/): Single-machine CPU core scaling, GPU benchmarks, and joint GPU+CPU training.
- [`early_smoke_runs/`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/archive/early_smoke_runs/): Preliminary smoke tests and setup verifications.
- [`legacy_logs/`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/results/archive/legacy_logs/): Historical nohup run logs.

