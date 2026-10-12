"""FP32 anchored Local SGD deltas over the existing FP16 MPI transports."""
import hashlib
import time
import numpy as np
from mpi4py import MPI
from src.training.fp16_communicator import FP16Communicator

DELTA_MODES = ('fp16_delta_allgather', 'fp16_delta_allreduce')


class FP16DeltaCommunicator(FP16Communicator):
    def __init__(self, comm, collective='allgather', verify_reference=False):
        super().__init__(comm, collective, verify_reference, log_dir=None)
        self.global_prev = None
        self._quantization_error = 0.0

    def set_anchor(self, flat):
        """Reject inconsistent or non-FP32 anchors collectively, without mutation."""
        valid = isinstance(flat, np.ndarray) and flat.dtype == np.float32 and flat.ndim == 1
        valid = valid and bool(np.isfinite(flat).all())
        if not self.comm.allreduce(valid, op=MPI.LAND):
            raise ValueError('Delta anchor must be finite one-dimensional FP32 on every rank')
        signature = (flat.shape, hashlib.sha256(flat.tobytes()).hexdigest())
        signatures = self.comm.allgather(signature)
        if any(s != signatures[0] for s in signatures):
            raise ValueError('global_prev differs across ranks')
        self.global_prev = flat.copy()

    def quantize(self, flat_w_fp32, factor=1.0):
        quantized, elapsed = super().quantize(flat_w_fp32, factor)
        # Undo pre-scaling for the local delta Eq; zero-weight ranks contribute
        # no delta and report zero error. Reduction roundoff is a separate error.
        target = flat_w_fp32.astype(np.float64)
        if factor > 0:
            recovered = quantized.astype(np.float64) / float(np.float32(factor))
            self._quantization_error = float(
                np.linalg.norm(target - recovered) / (np.linalg.norm(target) + 1e-12))
        else:
            self._quantization_error = 0.0
        return quantized, elapsed

    def synchronize(self, flat_w, samples_since_sync, policy='sample_weighted'):
        start = time.perf_counter()
        if policy not in ('sample_weighted', 'uniform'):
            raise ValueError('Unsupported averaging policy')
        if self.global_prev is None:
            raise RuntimeError('Initialize a shared FP32 anchor before local training')
        valid = flat_w.dtype == np.float32 and flat_w.shape == self.global_prev.shape
        valid = valid and bool(np.isfinite(flat_w).all())
        if not self.comm.allreduce(valid, op=MPI.LAND):
            raise ValueError('Local delta model must be finite FP32 and match the anchor')
        self.set_anchor(self.global_prev)
        t_encode = time.perf_counter()
        with np.errstate(over='ignore', invalid='ignore'):
            delta = flat_w - self.global_prev
        delta_ms = (time.perf_counter() - t_encode) * 1000.0
        averaged_delta, total, stats = super().synchronize(delta, samples_since_sync, policy)
        t_decode = time.perf_counter()
        with np.errstate(over='ignore', invalid='ignore'):
            result = self.global_prev + averaged_delta
        reconstruct_ms = (time.perf_counter() - t_decode) * 1000.0
        if not self.comm.allreduce(bool(np.isfinite(result).all()), op=MPI.LAND):
            raise FloatingPointError('Non-finite reconstructed global model')
        # Exact digest consensus also protects the anchor for the next round.
        self.set_anchor(result)
        stats.update(
            method='fp16_delta_' + self.collective,
            reduction_dtype='fp16' if self.collective == 'allreduce' else 'fp32',
            encode_ms=delta_ms + stats['cast_to_fp16_ms'],
            decode_ms=reconstruct_ms + stats['cast_accumulation_ms'],
            sync_total_ms=(time.perf_counter() - start) * 1000.0,
            relative_l2_quantization_error=self._quantization_error,
            delta_l2_norm=float(np.linalg.norm(delta.astype(np.float64))),
        )
        return result, total, stats
