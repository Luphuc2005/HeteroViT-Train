"""Run bounded FP32 training with whole/delta FP16 shadows at each model sync.

Invoke through sweep_fp16_comparison.sh --fp16-diagnostic. Diagnostic timings
contain extra collectives and must not be used to claim training speedup.
"""
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Let the normal entrypoint configure CUDA visibility, CPU threading and TF
# before installing the opt-in hook. Importing MPITrainer here would import TF
# too early and could change device initialization compared with the control.
state = runpy.run_path(str(ROOT / 'train_mpi.py'), run_name='__main__')

trainer = state['trainer']
rows = trainer.fp16_shadow_rows
iterations = int(trainer.optimizer.iterations.numpy())
valid = bool(rows) and iterations == trainer.max_steps
if not trainer.comm.allreduce(valid, op=state['MPI'].LAND):
    raise RuntimeError('Diagnostic did not reach max-steps on every rank')
if trainer.rank == 0:
    summary = dict(
        applied_training_method='fp32', shadow_collective='allgather',
        timing_scope='diagnostic_only_extra_collectives_no_speedup_claim',
        seed=int(state['config']['seed']), optimizer_steps_per_rank=iterations,
        synchronization_rounds=len(rows),
        delta_lower_error_rounds=sum(r['delta_lower_error'] for r in rows),
        equal_error_rounds=sum(r['delta_error_l2'] == r['whole_error_l2'] for r in rows),
    )
    for name in ('whole', 'delta'):
        for metric in ('error_l2', 'error_relative_to_update', 'max_absolute_error'):
            key = name + '_' + metric
            summary['max_' + key] = max(r[key] for r in rows)
            summary['mean_' + key] = sum(r[key] for r in rows) / len(rows)
    path = Path(trainer.logger.run_dir) / 'fp16_same_inputs_summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    print('FP16_SAME_INPUTS ' + json.dumps(summary), flush=True)
