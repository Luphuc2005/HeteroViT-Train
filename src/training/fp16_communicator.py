"""
FP16 Weight Communicator for Distributed Training.

Encapsulates all quantization, de-quantization, MPI collective operations,
overflow protection, and telemetry logging for FP16 communication in Local SGD.
Ensures local models and optimizer states remain strictly FP32.
"""
import time
import os
import csv
from typing import Dict, Any, Tuple, Optional
import numpy as np
from mpi4py import MPI


class FP16Communicator:
    """Handles FP16 weight quantization and communication across MPI ranks."""

    def __init__(
        self,
        comm: MPI.Intracomm,
        collective: str = "allreduce",
        verify_reference: bool = False,
        log_dir: Optional[str] = None,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.world_size = comm.Get_size()
        self.collective = str(collective).lower()
        self.verify_reference = verify_reference
        self.log_dir = log_dir
        self._custom_op = None

        if self.collective not in ("allreduce", "allgather"):
            raise ValueError(f"Unsupported collective: '{self.collective}'. Choose 'allreduce' or 'allgather'.")

        if self.collective == "allreduce":
            self._custom_op = MPI.Op.Create(self._fp16_sum_callback, commute=True)

    @staticmethod
    def _fp16_sum_callback(invec, inoutvec, datatype):
        """Custom MPI reduction callback for IEEE 754 half-precision floats."""
        a = np.frombuffer(invec, dtype=np.float16)
        b = np.frombuffer(inoutvec, dtype=np.float16)
        b[:] += a

    def __del__(self):
        if getattr(self, "_custom_op", None) is not None:
            try:
                self._custom_op.Free()
            except Exception:
                pass
            self._custom_op = None

    def quantize(self, flat_w_fp32: np.ndarray, factor: float = 1.0) -> Tuple[np.ndarray, float]:
        """Quantizes FP32 weights to FP16 with optional pre-scaling.
        
        Args:
            flat_w_fp32: 1D array of model weights in float32
            factor: Optional scaling factor (e.g. sample weight ratio p_i)
            
        Returns:
            (quantized_fp16, cast_time_ms)
        """
        t0 = time.perf_counter()
        with np.errstate(over="ignore", invalid="ignore"):
            if factor != 1.0:
                scaled = flat_w_fp32 * np.float32(factor)
                quantized = scaled.astype(np.float16)
            else:
                quantized = flat_w_fp32.astype(np.float16)
        cast_time_ms = (time.perf_counter() - t0) * 1000.0
        return quantized, cast_time_ms

    def dequantize(self, flat_w_fp16: np.ndarray) -> Tuple[np.ndarray, float]:
        """Dequantizes FP16 weights back to FP32.
        
        Args:
            flat_w_fp16: 1D array of weights in float16
            
        Returns:
            (recovered_fp32, cast_time_ms)
        """
        t0 = time.perf_counter()
        recovered = flat_w_fp16.astype(np.float32)
        cast_time_ms = (time.perf_counter() - t0) * 1000.0
        return recovered, cast_time_ms

    def synchronize(
        self,
        flat_w: np.ndarray,
        samples_since_sync: int,
        policy: str = "sample_weighted",
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Executes FP16 weight synchronization across ranks according to the collective method.
        
        Args:
            flat_w: Flattened 1D array of local model weights (float32)
            samples_since_sync: Local samples processed since last sync
            policy: Averaging policy ('sample_weighted' or 'uniform')
            
        Returns:
            (global_flat_w, total_samples, stats_dict)
        """
        stats: Dict[str, Any] = {
            "communication_dtype": "fp16",
            "collective": self.collective,
            "num_elements": int(flat_w.size),
            "payload_bytes": int(flat_w.size * 2),
            "cast_to_fp16_ms": 0.0,
            "communication_ms": 0.0,
            "cast_accumulation_ms": 0.0,
        }

        # Reference verification against ground-truth FP32 if requested
        reference_w = None
        if self.verify_reference:
            reference_w, _, _ = self._compute_fp32_reference(flat_w, samples_since_sync, policy)

        if self.collective == "allreduce":
            global_w, total_samples = self._sync_allreduce(flat_w, samples_since_sync, policy, stats)
        else:
            global_w, total_samples = self._sync_allgather(flat_w, samples_since_sync, policy, stats)

        # Validation: Ensure finite results and compute error vs reference if enabled
        valid = bool(np.isfinite(global_w).all())
        if not self.comm.allreduce(valid, op=MPI.LAND):
            raise FloatingPointError("Non-finite weights encountered after FP16 synchronization")

        if reference_w is not None:
            err = np.abs(global_w - reference_w)
            norm_ref = max(float(np.linalg.norm(reference_w)), 1e-30)
            stats.update(
                max_absolute_error=float(err.max()),
                mean_absolute_error=float(err.mean()),
                relative_l2_error=float(np.linalg.norm(err) / norm_ref),
            )

        # Write per-rank communication CSV if log_dir configured
        if self.log_dir:
            self._log_telemetry(stats)

        return global_w, total_samples, stats

    def _sync_allreduce(
        self,
        flat_w: np.ndarray,
        samples: int,
        policy: str,
        stats: Dict[str, Any],
    ) -> Tuple[np.ndarray, float]:
        """FP16 Ring Allreduce with normalized weights & IEEE 754 float16 reduction."""
        counts = self.comm.allgather(float(max(0, samples)))
        total = sum(counts) if policy == "sample_weighted" else float(self.world_size)
        if total <= 0:
            total = float(self.world_size)
            factor = 1.0 / float(self.world_size)
        else:
            rank_samples = float(max(0, samples)) if policy == "sample_weighted" else 1.0
            factor = rank_samples / total

        # Step 1: Quantize (FP32 -> FP16) with normalized sample factor
        send_fp16, cast_down_ms = self.quantize(flat_w, factor=factor)
        stats["cast_to_fp16_ms"] = cast_down_ms

        valid = self.comm.allreduce(bool(np.isfinite(send_fp16).all()), op=MPI.LAND)
        if not valid:
            raise FloatingPointError(f"Non-finite FP16 weights on rank {self.rank}")

        # Step 2: MPI Collective Allreduce using custom float16 sum
        t_comm = time.perf_counter()
        recv_fp16 = np.empty_like(send_fp16)
        self.comm.Allreduce([send_fp16, MPI.SHORT], [recv_fp16, MPI.SHORT], op=self._custom_op)
        stats["communication_ms"] = (time.perf_counter() - t_comm) * 1000.0

        # Step 3: Dequantize (FP16 -> FP32)
        global_w, cast_up_ms = self.dequantize(recv_fp16)
        stats["cast_accumulation_ms"] = cast_up_ms

        return global_w, total

    def _sync_allgather(
        self,
        flat_w: np.ndarray,
        samples: int,
        policy: str,
        stats: Dict[str, Any],
    ) -> Tuple[np.ndarray, float]:
        """FP16 Allgather via byte buffers with local FP32 accumulation."""
        # Step 1: Quantize (FP32 -> FP16) unscaled
        send_fp16, cast_down_ms = self.quantize(flat_w)
        stats["cast_to_fp16_ms"] = cast_down_ms

        valid = self.comm.allreduce(bool(np.isfinite(send_fp16).all()), op=MPI.LAND)
        if not valid:
            raise FloatingPointError(f"Non-finite FP16 weights on rank {self.rank}")

        # Step 2: MPI Allgather raw bytes
        counts = self.comm.allgather(float(max(0, samples)))
        received = np.empty((self.world_size, flat_w.size), dtype=np.float16)

        t_comm = time.perf_counter()
        self.comm.Allgather(
            [send_fp16.view(np.uint8), MPI.BYTE],
            [received.view(np.uint8), MPI.BYTE],
        )
        stats["communication_ms"] = (time.perf_counter() - t_comm) * 1000.0

        # Step 3: Dequantize & Accumulate in FP32
        t_up = time.perf_counter()
        total = sum(counts) if policy == "sample_weighted" else float(self.world_size)
        factors = counts if policy == "sample_weighted" and total > 0 else [1.0] * self.world_size
        if total <= 0:
            total = float(self.world_size)

        global_w = np.zeros_like(flat_w)
        for row, n in zip(received, factors):
            decoded, _ = self.dequantize(row)
            decoded *= np.float32(n)
            global_w += decoded
        global_w /= np.float32(total)
        stats["cast_accumulation_ms"] = (time.perf_counter() - t_up) * 1000.0

        return global_w, total

    def _compute_fp32_reference(
        self,
        flat_w: np.ndarray,
        samples: int,
        policy: str,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Computes reference FP32 Allreduce ground truth for verification."""
        n = float(max(0, samples))
        send = flat_w * n if policy == "sample_weighted" else flat_w
        result = np.empty_like(flat_w)
        self.comm.Allreduce(send, result, op=MPI.SUM)
        total = float(self.comm.allreduce(n, op=MPI.SUM)) if policy == "sample_weighted" else float(self.world_size)
        if total <= 0:
            self.comm.Allreduce(flat_w, result, op=MPI.SUM)
            total = float(self.world_size)
        result /= total
        return result, total, {}

    def _log_telemetry(self, stats: Dict[str, Any]):
        """Logs communication statistics to weight_communication_rank{rank}.csv."""
        path = os.path.join(self.log_dir, f"weight_communication_rank{self.rank}.csv")
        exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats))
            if not exists:
                writer.writeheader()
            writer.writerow(stats)
