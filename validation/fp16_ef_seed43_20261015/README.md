# FP16 Delta AllGather error-feedback smoke, seed 43

Both methods passed real-cluster smoke on lab01–lab05 with the existing
host-specific Python wrapper, 2 GPUs on rank 0 and CPUs on ranks 1–4.
Structured evidence is in [summary.json](summary.json).

| Check | Delta FP16 | Delta FP16 + EF |
|---|---:|---:|
| Optimizer steps per rank | 120 | 120 |
| Synchronizations per rank | 3 | 3 |
| Payload bytes per rank/sync | 5,387,156 | 5,387,156 |
| FP32 finite model/optimizer state | PASS | PASS |
| Final weights equal anchor, hashes identical across ranks | PASS | PASS |
| Per-rank FP32 residual finite and matching shape | N/A | PASS |
| Five collected state snapshots match last training model hashes | N/A | PASS |

EF retains 10,774,312 bytes of residual per rank. Residuals are local and need
not agree across ranks. Final snapshots contain communication state only;
they do not provide a complete optimizer/RNG/dataloader training resume.

Raw logs, reports and snapshots are local under
`results/fp16_sweep_seed43_5jdeOPhh/`. They are not committed here. This is a
120-step correctness check, not evidence of improved full-training accuracy
or speed. The subsequent 40-epoch seed-43 comparison is saved separately in
[full_comparison.json](full_comparison.json): Delta FP16 obtained 65.86% test
accuracy, versus 65.34% with EF. Both completed with 160 synchronizations/rank.
This single paired seed does not support an EF accuracy improvement.

The local MPI regression suite passed 18 transport tests/rank (including 6 EF
tests), and 3 diagnostic tests/rank on 5 processes. The 3 summary-parser tests,
matched-config validation, shell/Python syntax checks and dry runs also passed.
An additional one-rank malformed-state case was checked after cluster smoke;
the change only broadens collective rejection of corrupt snapshot metadata.

Reproduce smoke from the workspace root:

```bash
bash HeteroViT-MPI/scripts/run_fp16_followup.sh --phase smoke
```

Run the full seed-43 pair separately:

```bash
nohup bash HeteroViT-MPI/scripts/run_fp16_followup.sh --phase full \
  > HeteroViT-MPI/fp16_ef_seed43_40e.log 2>&1 &
```
