"""Weight transport only; all arithmetic and model state remain FP32."""
import time
import numpy as np
from mpi4py import MPI
from src.training.fp16_communicator import FP16Communicator


def average_weights(comm, flat, samples, policy='sample_weighted', precision='fp32', collective=None, global_prev=None):
    from src.training.fp16_delta_communicator import DELTA_MODES, FP16DeltaCommunicator
    if precision in DELTA_MODES:
        transport = FP16DeltaCommunicator(comm, collective=precision[len('fp16_delta_'):])
        transport.set_anchor(global_prev)
        return transport.synchronize(flat, samples, policy)

    if precision not in ('fp32', 'fp16') or policy not in ('sample_weighted', 'uniform'):
        raise ValueError('Unsupported communication precision or averaging policy')

    if collective is None:
        collective = 'allreduce' if precision == 'fp32' else 'allgather'
    collective = str(collective).lower()
    if collective not in ('allreduce', 'allgather'):
        raise ValueError(f"Unsupported collective: '{collective}'. Choose 'allreduce' or 'allgather'.")

    if precision == 'fp16':
        communicator = FP16Communicator(comm=comm, collective=collective)
        return communicator.synchronize(flat_w=flat, samples_since_sync=samples, policy=policy)

    # Standard FP32 baseline
    stats = dict(
        communication_dtype='fp32',
        collective='allreduce',
        num_elements=int(flat.size),
        payload_bytes=int(flat.size * 4),
        cast_to_fp16_ms=0.0,
        communication_ms=0.0,
        cast_accumulation_ms=0.0,
    )
    start = time.perf_counter()
    n = float(max(0, samples))
    send = flat * n if policy == 'sample_weighted' else flat
    result = np.empty_like(flat)
    comm.Allreduce(send, result, op=MPI.SUM)
    total = float(comm.allreduce(n, op=MPI.SUM)) if policy == 'sample_weighted' else float(comm.Get_size())
    if total <= 0:
        comm.Allreduce(flat, result, op=MPI.SUM)
        total = float(comm.Get_size())
    result /= total
    stats['communication_ms'] = (time.perf_counter() - start) * 1000.0
    return result, total, stats
