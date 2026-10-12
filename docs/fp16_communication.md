# FP16 weight communication baseline

Set `training.comm_precision: fp16` for Local SGD. Default `fp32` preserves
weighted Allreduce. Training, optimizer and model weights remain FP32.
FP16 uses Allgather with explicit MPI.BYTE buffers, then decodes each rank's
weights to FP32 before multiplying by processed sample counts. Zero total
samples retains the existing uniform fallback. Overflow is rejected collectively.

Allgather holds world_size FP16 models on each rank plus FP32 accumulation
scratch space. Halving each rank's model payload does not imply halving network
traffic or improving speed: collective algorithms differ from Allreduce.

Each rank writes `weight_communication_rankN.csv`. Payload means one rank's
model buffer, excluding metadata and diagnostics, not measured wire traffic.
Communication time includes averaging collectives and sample-count metadata;
FP16 cast/accumulation are separate. Total sync includes flattening, diagnostics,
model assignment and optional verification; excludes CSV/log output. Sum times
per rank, then report the maximum rank total for comparison. Reference checking
adds FP32 traffic and is for smoke only.

Transport check (one host, five processes):

```bash
mpiexec --oversubscribe -n 5 python HeteroViT-MPI/tools/smoke_weight_communication.py
```

Real five-node ViT smoke from repository root, using the existing launcher:

```bash
mpi_cluster/run_5nodes.sh --sync --config HeteroViT-MPI/configs/mpi/local_sgd_h40_fp16_smoke.yaml --max-steps 120 --seed 42
```

Require three H=40 synchronization rounds, finite training losses and weights,
continued optimizer updates, and successful per-round rank consistency checks.
Review max/mean absolute and relative L2 errors in communication CSVs. The smoke
config enables FP32 reference comparison. Do not use it for performance results.

Only after the real training smoke passes, run the two 40-epoch configs:
`local_sgd_h40_fp32_40e.yaml` and `local_sgd_h40_fp16_40e.yaml`, using the same
launcher and hardware. Both fix seed 42, H=40, batch [266,6,6,6,16], model,
optimizer and learning rate. Use distinct experiment output directories.
Report test accuracy and total training time from experiment logs, communication
and total sync time from the per-rank CSVs, per-rank payload bytes, and summed
cast-to-FP16 plus cast/accumulation time. Include collective type with results.

Validation performed: syntax compilation and local five-rank transport smoke
passed, including 3 rounds of 40 synthetic FP32 updates, weighted/uniform and
zero-sample averages, identical ranks and a weighting-before-cast overflow case.
Maximum observed absolute error versus FP32 was 1.1143e-4; mean absolute error
1.1010e-5 to 1.2626e-5; relative L2 1.8048e-4 to 2.0648e-4.
These numbers are synthetic, not ViT/CIFAR-10 experiment results.
Real training smoke and 40 epochs have not run: the available Python fails to
import TensorFlow because `six` is missing. No full experiment is certified.
