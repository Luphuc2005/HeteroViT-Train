"""FP16 delta AllGather with per-rank FP32 quantization error feedback."""
import time
from zipfile import BadZipFile
import numpy as np
from mpi4py import MPI

from src.training.fp16_delta_communicator import FP16DeltaCommunicator


class FP16DeltaEFCommunicator(FP16DeltaCommunicator):
    """Carry raw encode/decode error between rounds with fixed sample ratios.

    Residuals compensate local FP16 conversion only. They do not compensate
    FP32 accumulation/reconstruction or local optimizer drift.
    """

    def __init__(self, comm, collective='allgather', verify_reference=False):
        if collective != 'allgather':
            raise ValueError('Delta error feedback supports AllGather only')
        super().__init__(comm, collective, verify_reference)
        self.residual = None
        self._pending_residual = None
        self._sample_factors = None
        self._feedback_ms = 0.0

    def set_anchor(self, flat):
        super().set_anchor(flat)
        if self.residual is None:
            self.residual = np.zeros_like(flat)

    def quantize(self, delta, factor=1.0):
        if factor != 1.0:
            raise ValueError('Error feedback requires unscaled AllGather payloads')
        start = time.perf_counter()
        with np.errstate(over='ignore', invalid='ignore'):
            corrected = delta + self.residual
        self._feedback_ms = (time.perf_counter() - start) * 1000.0
        # Reuse exactly the original FP16 casting and Eq calculation. Eq now
        # refers to corrected delta; the raw delta norm remains in parent stats.
        quantized, cast_ms = super().quantize(corrected, factor)
        start = time.perf_counter()
        with np.errstate(over='ignore', invalid='ignore'):
            self._pending_residual = corrected - quantized.astype(np.float32)
        self._feedback_ms += (time.perf_counter() - start) * 1000.0
        return quantized, cast_ms

    def synchronize(self, flat_w, samples_since_sync, policy='sample_weighted'):
        start = time.perf_counter()
        if self.global_prev is None:
            raise RuntimeError('Initialize a shared FP32 anchor before local training')
        valid = (isinstance(self.residual, np.ndarray)
                 and self.residual.dtype == np.float32
                 and self.residual.shape == self.global_prev.shape
                 and bool(np.isfinite(self.residual).all()))
        if not self.comm.allreduce(valid, op=MPI.LAND):
            raise ValueError('Error-feedback residual must be finite matching FP32')
        if policy not in ('sample_weighted', 'uniform'):
            raise ValueError('Unsupported averaging policy')
        counts = self.comm.allgather(float(max(0, samples_since_sync)))
        total = sum(counts)
        factors = (np.asarray(counts, dtype=np.float64) / total
                   if policy == 'sample_weighted' and total > 0
                   else np.full(self.world_size, 1.0 / self.world_size))
        # Raw residuals telescope after weighted averaging only when relative
        # sample weights remain fixed. Forced epoch-end syncs are supported.
        valid = (self._sample_factors is None or
                 bool(np.allclose(factors, self._sample_factors, rtol=0, atol=1e-12)))
        if not self.comm.allreduce(valid, op=MPI.LAND):
            raise ValueError('Error feedback requires fixed relative sample weights')
        self._pending_residual = None
        result, total, stats = super().synchronize(flat_w, samples_since_sync, policy)
        # Commit memory only after parent finite/consensus checks succeed.
        self.residual = self._pending_residual
        self._pending_residual = None
        self._sample_factors = factors.copy()
        stats.update(
            method='fp16_delta_allgather_ef', error_feedback=True,
            residual_dtype='float32', residual_bytes=int(self.residual.nbytes),
            residual_l2_norm=float(np.linalg.norm(self.residual.astype(np.float64))),
            error_feedback_ms=self._feedback_ms,
            quantization_error_target='delta_plus_residual',
        )
        stats['encode_ms'] += self._feedback_ms
        stats['sync_total_ms'] = (time.perf_counter() - start) * 1000.0
        return result, total, stats

    def save_state(self, path):
        """Save communication state only, not optimizer/RNG/dataloader state."""
        if self.residual is None:
            raise RuntimeError('Initialize anchor before saving feedback state')
        np.savez(path, version=np.int64(1), rank=np.int64(self.rank),
                 world_size=np.int64(self.world_size), anchor=self.global_prev,
                 residual=self.residual,
                 sample_factors=(self._sample_factors if self._sample_factors is not None
                                 else np.empty(0, dtype=np.float64)))

    def load_state(self, path):
        """Collectively validate a per-rank communication-state snapshot."""
        valid = True
        try:
            with np.load(path, allow_pickle=False) as saved:
                anchor = saved['anchor'].copy()
                residual = saved['residual'].copy()
                factors = saved['sample_factors'].copy()
                valid = (int(saved['version']) == 1 and int(saved['rank']) == self.rank
                         and int(saved['world_size']) == self.world_size
                         and anchor.dtype == residual.dtype == np.float32
                         and anchor.ndim == 1 and anchor.shape == residual.shape
                         and np.isfinite(anchor).all() and np.isfinite(residual).all()
                         and factors.dtype == np.float64
                         and factors.shape in ((0,), (self.world_size,))
                         and np.isfinite(factors).all() and (factors >= 0).all()
                         and (factors.size == 0 or abs(float(factors.sum()) - 1) < 1e-12))
        except (OSError, ValueError, KeyError, TypeError, OverflowError, BadZipFile):
            valid = False
        if not self.comm.allreduce(bool(valid), op=MPI.LAND):
            raise ValueError('Invalid per-rank error-feedback state')
        signatures = self.comm.allgather(tuple(factors))
        if any(s != signatures[0] for s in signatures):
            raise ValueError('Sample weights differ across feedback snapshots')
        self.set_anchor(anchor)
        self.residual = residual
        self._sample_factors = factors if factors.size else None
