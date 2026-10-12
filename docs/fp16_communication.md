# FP16 weight communication baseline

Set `training.comm_precision: fp16` for Local SGD. Default `fp32` preserves
weighted Allreduce. Training, optimizer and model weights remain FP32.
With `comm_collective: allgather`, FP16 uses explicit MPI.BYTE buffers, then
decodes each rank's weights to FP32 before multiplying by processed sample counts.
With `comm_collective: allreduce`, it pre-scales in FP32 and uses the custom
FP16 reduction callback; the trainer default is Allreduce. Zero total
samples retains the existing uniform fallback. Overflow is rejected collectively.

Allgather holds world_size FP16 models on each rank plus FP32 accumulation
scratch space. Halving each rank's model payload does not imply halving network
traffic or improving speed: collective algorithms differ from Allreduce.

Each rank writes `weight_communication_rankN.csv`. Payload means one rank's
model buffer, excluding metadata and diagnostics, not measured wire traffic.
Communication time is the FP16 payload collective time (FP32 baseline also
includes sample-count metadata);
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
At the time of the original whole-weight validation, real training smoke and
40 epochs had not run: the default Python lacked `six`. The later delta cluster
validation below uses the launcher environments, which import TensorFlow
successfully. No full 40-epoch experiment is certified.

## FP16 Delta Local SGD (H=40)

New opt-in values for `training.comm_precision` / `--comm-precision`:
`fp16_delta_allgather` and `fp16_delta_allreduce`. The mode selects its collective.
Existing `fp32`, `fp16` plus `comm_collective`, baseline configs and MPI Ring
runtime settings are unchanged. The existing whole-weight FP16 YAML currently
selects **allreduce**, despite older documentation implying an Allgather default.

`FP16DeltaCommunicator` reuses the original FP16 transport methods unchanged.
Before the first local update, the trainer broadcasts rank 0's FP32 model and
stores an independent FP32 anchor (also applies after checkpoint restoration).
Every synchronization checks anchor SHA256 digests across ranks, subtracts that
anchor in FP32, communicates deltas, and adds the averaged delta in FP32.
The reconstructed result is checked for finiteness and exact digest consensus
before becoming the next anchor. Neither caller buffers nor optimizer state
are cast to FP16. Delta training rejects non-FP32 model/compute/optimizer floating
state; integer optimizer iteration counters are allowed.

Allgather sends unscaled FP16 deltas via MPI.BYTE, decodes each rank to FP32,
multiplies by sample counts, accumulates and divides in FP32. Allreduce uses the
existing normalized `p_i * delta` FP32 pre-scaling followed by FP16 casting and
MPI.SHORT with the existing custom callback: **the reduction is still FP16**.
It is not an FP32 accumulation of FP16 inputs. Zero total samples falls back to
uniform averaging in both modes. Invalid inputs/cast overflow/reduction overflow
and non-finite reconstruction fail collectively without updating the anchor.

Delta modes write exactly one row per synchronization per rank through the
trainer's `weight_communication_rankN.csv`. Fields include `method`, `collective`,
`reduction_dtype`, `encode_ms`, `communication_ms`, `decode_ms`, `sync_total_ms`,
`payload_bytes`, `relative_l2_quantization_error`, and `delta_l2_norm`.
`encode_ms` measures FP32 subtraction plus FP16 cast; `decode_ms` measures decode,
averaging and reconstruction. `communication_ms` is the existing transport's
payload collective time (excludes count/validation/digest metadata).
`sync_total_ms` includes flattening, count/validation/digest metadata, diagnostics,
optional reference checking and model assignment, excluding CSV/log output.
Quantization diagnostic norms are computed in FP64 to avoid metric overflow;
model and averaging arithmetic remain FP32 except the Allreduce callback.
Payload is `2 * number_of_model_elements` per rank, not measured wire traffic.
Delta sends the same dense number of elements as whole weights; no sparsification.

Eq is a per-rank local cast error, not global reduction error. For Allgather,
`hat_delta = float32(float16(delta))`. For Allreduce with nonzero `p_i`,
`hat_delta = float32(float16(p_i * delta)) / p_i`; Eq compares that to delta.
Zero-weight ranks report zero Eq because their contribution is zero.
`delta_l2_norm` always measures the unscaled local delta. Optional
`comm_verify_reference: true` additionally compares the averaged delta against
FP32 averaging (`relative_l2_error`); this includes FP16 reduction roundoff and
excludes anchor addition. It adds traffic and must be off for timing experiments.

From `HeteroViT-MPI`, run the five-rank transport regression/smoke suite:

```bash
mpiexec --oversubscribe -n 5 python -m unittest discover -s tests -p 'test_*comm*.py'
```

For the verified five-node delta workflow, use the isolated staging commands
in `validation/delta_20261012/README.md`. They preserve the existing launcher
hostfile, Python selection and MPI flags while avoiding its destructive
`--sync` cleanup. The two new 40-epoch configs fix seed 42, H=40, batches,
optimizer and learning rate; the separate smoke configs enable reference checks.

For whole-weight comparisons use the existing FP16 config with
`comm_collective: allgather` / `allreduce` in separate copied configs; preserve
original baseline configs. Compare with `local_sgd_h40_fp32_40e.yaml`.
Use distinct output directories and compare maximum rank summed sync times,
accuracy and training time. Existing whole-weight logging writes twice per sync
with inconsistent CSV schemas (communicator and trainer both append); that
pre-existing behavior is preserved here, so do not blindly sum its raw CSV rows.

Initial validation on 2026-10-12: 12 tests per rank passed with five local MPI ranks,
covering existing baselines, multiple delta rounds, exact zero delta, FP32
reference tolerance, exact Allgather sample-weighted decoding, uniform/zero-count
fallback, identical reconstructed ranks, anchor mismatch and collective
NaN/Inf/cast-overflow rejection. Python syntax compilation passed.
The initial default-Python TensorFlow import failed because `six` was missing.
Subsequent launcher-runtime checks succeeded; see the cluster validation below.
No 40-epoch accuracy/performance claims.


## Cluster runtime and delta training verification (2026-10-12)

`tools/preflight_delta_runtime.py` verified the actual launcher environments:
all ranks use Python 3.8.10, NumPy 1.23.5 and mpi4py 4.1.1. TensorFlow is 2.10.1
on lab01/lab05 and 2.10.0 on lab02/lab03/lab04. OpenMPI is 4.0.3 on lab01 and
4.0.2 on workers. All ranks have identical ordered model layouts (2,693,578
parameters), FP32 compute/weights and experimental Keras AdamW floating state.
These versions differ from system/default Python versions reported separately.

Validation runs use an isolated copy of `src`, `configs`, `tests`, `tools` and
`train_mpi.py` at `/tmp/heterovit_delta_validation_20261012` on each node. They
use the same hostfile, host-specific Python wrapper and MPI network flags as
the existing launcher. The original worker repositories and existing jobs are
preserved. Do not use `run_5nodes.sh --sync` for this workflow: that option
kills worker Python processes and uses `rsync --delete`.

Two dedicated smoke configs enable FP32 reference and per-round consistency
checks: `local_sgd_h40_fp16_delta_allgather_smoke.yaml` and
`local_sgd_h40_fp16_delta_allreduce_smoke.yaml` (one epoch, max 120 steps).
`tools/smoke_delta_training.py` runs the unchanged training entry and then
checks optimizer iteration 120 on every rank, finite FP32 model and optimizer
floating state, equality with the stored anchor, and identical final model
SHA256 across ranks. It adds no optimizer updates.

Logs, runtime reports and per-rank CSV copies are saved under
`validation/delta_20261012`; see its README and summary.json for measured
results. Raw logs/runtime reports/CSV are local artifacts; Git retains only the
compact README, summary and summary script. Reference-enabled smoke timings
are diagnostic, not performance
measurements. Neither full 40-epoch experiments nor test-accuracy comparisons
have been run by this validation workflow.

## One-command five-method sweep

From the workspace root:

```bash
nohup bash HeteroViT-MPI/scripts/sweep_fp16_comparison.sh > HeteroViT-MPI/sweep_fp16_comparison.log 2>&1 &
```

Default order: FP32, FP16 whole-weight AllGather, FP16 whole-weight AllReduce,
FP16 delta AllGather, FP16 delta AllReduce. Each runs 40 epochs, seed 42, H=40,
batch `[266,6,6,6,16]`, sample-weighted averaging, matching model/optimizer/LR.
Reference checking is disabled in training. The added whole-weight AllGather
config copies the original whole-weight baseline and changes only experiment
name and collective. Existing baseline files and arithmetic are preserved.

The script snapshots current source/configs and the hostfile/Python wrapper
into a unique staging directory on all nodes, then runs sequentially with the
existing MPI flags. It creates distinct result roots under
`results/fp16_sweep_XXXXXXXX/`, with five method logs, `runtime.json`, per-node
CSV copies, `status.txt`, `summary.csv` and `summary.json`. It stops on any
failed training, collection or summary command. It does not invoke the
launcher's `--sync` cleanup.

`tools/summarize_fp16_sweep.py` recognizes the pre-existing whole-weight CSV
schema: communicator-only rows are skipped; the extra final field of trainer
rows is decoded as `total_sync_ms`. Timing is summed per rank before taking
the maximum. Summary includes test accuracy, training duration, synchronization
count, communication/sync/encode/decode totals and payload per rank. FP32's
existing encode/decode counters are zero; delta includes subtraction and
reconstruction, while whole-weight counters cover cast/accumulation.

To inspect the plan without launching jobs, add `--dry-run`. Optional
`--epochs N` and `--max-steps N` support a shorter orchestration smoke; test
accuracy remains blank when the existing training entry skips test evaluation
for a max-steps run. No other training overrides are accepted by the script.
The script was checked with shell syntax validation, dry-run of all five
methods, matched-config validation, three telemetry parser regressions and the
ten saved real delta smoke CSVs. The full 40-epoch sweep has not been launched.
