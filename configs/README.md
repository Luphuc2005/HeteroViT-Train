# HeteroViT Configuration Directory Structure (`configs/`)

This directory houses all training configurations categorized by deployment target and experimental phase:

```
configs/
├── mpi/        # Distributed 5-Node Cluster Configurations (ACTIVE PRODUCTION)
│   ├── auto_zero_idle_300.yaml   # Clean Synchronous AllReduce Baseline (Batch 300)
│   ├── local_sgd_300.yaml        # Local SGD with configurable H (4, 8, 16) & sample weighting
│   ├── master_agg_300.yaml       # Master Aggregation topology
│   ├── archive/                  # Legacy and diagnostic cluster YAMLs
│   └── README.md                 # Detailed documentation for cluster MPI configs
│
├── joint/      # Single-Node Joint GPU+CPU Training (Phase 0)
│   └── joint_2gpu_b*.yaml        # Multi-device workstation experiments (2x Titan Z + local CPU)
│
├── gpu/        # Single-Node GPU Scaling Benchmarks (Phase 0)
│   └── gpu_*.yaml                # 1x V100, 2x Titan Z, core count sweeps
│
└── cpu/        # Single-Node CPU Scaling Benchmarks (Phase 0)
    └── cpu_*.yaml                # Core scaling (6, 12, 18, 24, 30, 36, 42 cores)
```

For the active 5-node distributed cluster benchmark suite, refer to [`configs/mpi/README.md`](file:///home/icip/Ha/HeteroViT-Project/HeteroViT-MPI/configs/mpi/README.md).
