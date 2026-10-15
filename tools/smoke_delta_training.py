"""Run the unchanged training entry, then collectively verify final delta state.

Use with a delta smoke config and --max-steps. Adds checks, not training updates.
"""
import hashlib
import json
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
state = runpy.run_path(str(root / 'train_mpi.py'), run_name='__main__')
trainer = state['trainer']
comm = state['comm']
np = state['np']
MPI = state['MPI']
weights = trainer.model.get_weights()
optimizer_variables = trainer.optimizer.variables()
flat = np.concatenate([w.ravel() for w in weights])
iterations = int(trainer.optimizer.iterations.numpy())
feedback = getattr(trainer, 'comm_error_feedback', False)
residual = trainer.delta_communicator.residual if feedback else None
residual_valid = (not feedback or (residual.dtype == np.float32
                  and residual.shape == flat.shape and np.isfinite(residual).all()))
valid = (trainer.delta_communicator is not None
         and iterations == trainer.max_steps
         and trainer.model.dtype_policy.compute_dtype == 'float32'
         and all(w.dtype == np.float32 and np.isfinite(w).all() for w in weights)
         and all(not v.dtype.is_floating or (
             v.dtype.as_numpy_dtype == np.float32 and np.isfinite(v.numpy()).all())
                 for v in optimizer_variables)
         and np.array_equal(flat, trainer.delta_communicator.global_prev)
         and residual_valid)
report = dict(rank=comm.rank, host=state['hostname'], run_dir=trainer.logger.run_dir,
              optimizer_iterations=iterations, optimizer_variables=len(optimizer_variables),
              optimizer_dtypes=sorted(set(v.dtype.name for v in optimizer_variables)),
              model_sha256=hashlib.sha256(flat.tobytes()).hexdigest(),
              error_feedback=feedback,
              residual_l2_norm=float(np.linalg.norm(residual.astype(np.float64))) if feedback else 0.0,
              valid=bool(valid))
reports = comm.allgather(report)
passed = all(r['valid'] for r in reports) and len(set(r['model_sha256'] for r in reports)) == 1
if comm.rank == 0:
    print('DELTA_SMOKE_STATE ' + json.dumps(dict(passed=passed, ranks=reports)), flush=True)
if not passed:
    raise RuntimeError('Delta smoke final state validation failed')
