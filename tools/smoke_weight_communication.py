"""Transport smoke (synthetic FP32 local updates); does not certify ViT training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpi4py import MPI
from src.training.weight_communication import average_weights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
assert comm.Get_size() == 5
rng = np.random.RandomState(42 + rank)
w = rng.normal(0, .1, 8192).astype(np.float32)
batches = [266, 6, 6, 6, 16]
for round_id in range(3):
    for _ in range(40):
        w -= np.float32(.001) * rng.normal(0, .01, w.size).astype(np.float32)
    reference, _, _ = average_weights(comm, w, batches[rank] * 40, precision='fp32')
    gathered = comm.allgather(w)
    for policy, samples in [('sample_weighted', batches[rank] * 40), ('uniform', 0), ('sample_weighted', 0)]:
        result, _, stats = average_weights(comm, w, samples, policy, 'fp16')
        factors = np.array(batches, dtype=np.float32) if samples else np.ones(5, np.float32)
        expected = np.zeros_like(w)
        for local, factor in zip(gathered, factors):
            expected += local.astype(np.float16).astype(np.float32) * factor
        expected /= factors.sum()
        assert result.dtype == np.float32 and np.isfinite(result).all()
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
        assert stats['payload_bytes'] == w.size * 2
        assert all(np.array_equal(result, peer) for peer in comm.allgather(result))
        if samples:
            error = np.abs(result - reference)
            if rank == 0:
                print(dict(round=round_id + 1, max_absolute_error=float(error.max()),
                           mean_absolute_error=float(error.mean()),
                           relative_l2_error=float(np.linalg.norm(error) / np.linalg.norm(reference))), flush=True)
            next_w = result
    w = next_w
# Weighting before casting would overflow these otherwise representable weights.
w.fill(1000)
result, _, _ = average_weights(comm, w, batches[rank] * 40, precision='fp16')
assert np.all(result == 1000)
if rank == 0:
    print('PASS: transport smoke, 3 rounds H=40; real training smoke still required', flush=True)
