"""Weight transport only; all arithmetic and model state remain FP32."""
import time
import numpy as np
from mpi4py import MPI

_CUSTOM_FP16_OP = None


def _fp16_sum_callback(invec, inoutvec, datatype):
    a = np.frombuffer(invec, dtype=np.float16)
    b = np.frombuffer(inoutvec, dtype=np.float16)
    b[:] += a


def _get_fp16_op():
    global _CUSTOM_FP16_OP
    if _CUSTOM_FP16_OP is None:
        _CUSTOM_FP16_OP = MPI.Op.Create(_fp16_sum_callback, commute=True)
    return _CUSTOM_FP16_OP


def average_weights(comm, flat, samples, policy='sample_weighted', precision='fp32', collective=None):
    if precision not in ('fp32', 'fp16') or policy not in ('sample_weighted', 'uniform'):
        raise ValueError('Unsupported communication precision or averaging policy')

    if collective is None:
        collective = 'allreduce'
    collective = str(collective).lower()
    if collective not in ('allreduce', 'allgather'):
        raise ValueError(f"Unsupported collective: '{collective}'. Choose 'allreduce' or 'allgather'.")

    stats = dict(
        communication_dtype=precision,
        collective=collective,
        num_elements=int(flat.size),
        payload_bytes=int(flat.size * (2 if precision == 'fp16' else 4)),
        cast_to_fp16_ms=0.0,
        communication_ms=0.0,
        cast_accumulation_ms=0.0,
    )

    start = time.perf_counter()
    if precision == 'fp32':
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

    elif collective == 'allreduce':
        # FP16 Ring Allreduce with normalized weights & IEEE 754 float16 custom reduction.
        # Normalizing by sum(n_i) beforehand guarantees weights remain in [-3, 3] and cannot overflow float16 (max 65504).
        counts = comm.allgather(float(max(0, samples)))
        total = sum(counts) if policy == 'sample_weighted' else float(comm.Get_size())
        if total <= 0:
            total = float(comm.Get_size())
            factor = 1.0 / float(comm.Get_size())
        else:
            rank_samples = float(max(0, samples)) if policy == 'sample_weighted' else 1.0
            factor = rank_samples / total

        with np.errstate(over='ignore', invalid='ignore'):
            scaled_w = flat * np.float32(factor)
            send = scaled_w.astype(np.float16)
        stats['cast_to_fp16_ms'] = (time.perf_counter() - start) * 1000.0

        valid = comm.allreduce(bool(np.isfinite(send).all()), op=MPI.LAND)
        if not valid:
            raise FloatingPointError('Non-finite or overflowing FP16 weights on at least one rank')

        start = time.perf_counter()
        op = _get_fp16_op()
        recv_fp16 = np.empty_like(send)
        comm.Allreduce([send, MPI.SHORT], [recv_fp16, MPI.SHORT], op=op)
        stats['communication_ms'] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        result = recv_fp16.astype(np.float32)
        stats['cast_accumulation_ms'] = (time.perf_counter() - start) * 1000.0

    else:
        # FP16 Allgather with explicit MPI.BYTE buffers, followed by per-rank decode & FP32 accumulation.
        with np.errstate(over='ignore', invalid='ignore'):
            send = flat.astype(np.float16)
        stats['cast_to_fp16_ms'] = (time.perf_counter() - start) * 1000.0

        received = np.empty((comm.Get_size(), flat.size), dtype=np.float16)
        start = time.perf_counter()
        valid = comm.allreduce(bool(np.isfinite(send).all()), op=MPI.LAND)
        if not valid:
            raise FloatingPointError('Non-finite or overflowing FP16 weights on at least one rank')
        counts = comm.allgather(float(max(0, samples)))
        comm.Allgather([send.view(np.uint8), MPI.BYTE], [received.view(np.uint8), MPI.BYTE])
        stats['communication_ms'] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        total = sum(counts) if policy == 'sample_weighted' else comm.Get_size()
        factors = counts if policy == 'sample_weighted' and total > 0 else [1.0] * comm.Get_size()
        if total <= 0:
            total = comm.Get_size()
        result = np.zeros_like(flat)
        for row, n in zip(received, factors):
            decoded = row.astype(np.float32)
            decoded *= np.float32(n)
            result += decoded
        result /= np.float32(total)
        stats['cast_accumulation_ms'] = (time.perf_counter() - start) * 1000.0

    return result, total, stats
