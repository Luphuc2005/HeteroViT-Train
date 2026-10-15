import tempfile
import unittest
from pathlib import Path

import numpy as np
from mpi4py import MPI

from src.training.fp16_delta_communicator import FP16DeltaCommunicator
from src.training.fp16_delta_ef_communicator import FP16DeltaEFCommunicator


class TestDeltaErrorFeedback(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.anchor = np.linspace(0.12345, 1.12345, 128, dtype=np.float32)

    def transport(self):
        result = FP16DeltaEFCommunicator(self.comm)
        result.set_anchor(self.anchor)
        return result

    def test_weighted_protocol_and_residual_across_rounds(self):
        transport = self.transport()
        for step, window in enumerate([40, 40, 30, 20]):
            previous = transport.global_prev.copy()
            memory = transport.residual.copy()
            local = previous + np.float32((self.rank + 1) * (step + 1) * 1e-5)
            original_local = local.copy()
            corrected = (local - previous) + memory
            decoded = corrected.astype(np.float16).astype(np.float32)
            expected_memory = corrected - decoded
            samples = (self.rank + 1) * window
            rows = self.comm.allgather(decoded)
            counts = self.comm.allgather(samples)
            average = np.zeros_like(previous)
            for row, count in zip(rows, counts):
                average += row * np.float32(count)
            average /= np.float32(sum(counts))
            result, total, stats = transport.synchronize(local, samples)
            np.testing.assert_array_equal(result, previous + average)
            np.testing.assert_array_equal(transport.global_prev, result)
            np.testing.assert_array_equal(transport.residual, expected_memory)
            np.testing.assert_array_equal(local, original_local)
            for other in self.comm.allgather(result):
                np.testing.assert_array_equal(other, result)
            self.assertEqual(total, sum(counts))
            self.assertEqual(transport.residual.dtype, np.float32)
            self.assertEqual(stats['method'], 'fp16_delta_allgather_ef')
            self.assertEqual(stats['payload_bytes'], local.size * 2)
            self.assertEqual(stats['residual_bytes'], local.nbytes)
            self.assertTrue(stats['error_feedback'])

    def test_feedback_recovers_repeated_subnormal_updates(self):
        zero = np.zeros(32, dtype=np.float32)
        plain = FP16DeltaCommunicator(self.comm, 'allgather')
        plain.set_anchor(zero)
        feedback = FP16DeltaEFCommunicator(self.comm)
        feedback.set_anchor(zero)
        cumulative = np.zeros_like(zero, dtype=np.float64)
        for _ in range(32):
            raw = feedback.global_prev + np.float32(1e-8)
            cumulative += (raw - feedback.global_prev).astype(np.float64)
            feedback.synchronize(raw, self.rank + 1)
            plain.synchronize(plain.global_prev + np.float32(1e-8), self.rank + 1)
        error_ef = np.linalg.norm(feedback.global_prev.astype(np.float64) - cumulative)
        error_plain = np.linalg.norm(plain.global_prev.astype(np.float64) - cumulative)
        self.assertLess(error_ef, error_plain * 0.1)

    def test_failure_preserves_committed_anchor_and_residual(self):
        transport = self.transport()
        transport.synchronize(self.anchor + np.float32((self.rank + 1) * 1e-5), self.rank + 1)
        previous = transport.global_prev.copy()
        memory = transport.residual.copy()
        for bad in (np.nan, np.inf, 1e6):
            local = previous.copy()
            if self.rank == 0:
                local[0] = bad
            with self.assertRaises((ValueError, FloatingPointError)):
                transport.synchronize(local, self.rank + 1)
            np.testing.assert_array_equal(transport.global_prev, previous)
            np.testing.assert_array_equal(transport.residual, memory)
        result, _, _ = transport.synchronize(previous + np.float32(1e-5), self.rank + 1)
        self.assertTrue(np.isfinite(result).all())

    def test_changing_relative_sample_weights_is_rejected(self):
        if self.comm.size < 2:
            self.skipTest('Requires multiple ranks')
        transport = self.transport()
        transport.synchronize(self.anchor.copy(), self.rank + 1)
        memory = transport.residual.copy()
        with self.assertRaisesRegex(ValueError, 'fixed relative'):
            transport.synchronize(self.anchor.copy(), self.rank + 2)
        np.testing.assert_array_equal(transport.residual, memory)

    def test_state_roundtrip_continues_feedback_and_rejects_missing_rank_file(self):
        transport = self.transport()
        transport.synchronize(self.anchor + np.float32((self.rank + 1) * 1e-5), self.rank + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feedback.npz'
            transport.save_state(path)
            restored = FP16DeltaEFCommunicator(self.comm)
            restored.load_state(path)
            np.testing.assert_array_equal(restored.global_prev, transport.global_prev)
            np.testing.assert_array_equal(restored.residual, transport.residual)
            local = transport.global_prev + np.float32((self.rank + 1) * 2e-5)
            expected, _, _ = transport.synchronize(local, self.rank + 1)
            actual, _, _ = restored.synchronize(local, self.rank + 1)
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(restored.residual, transport.residual)
            before = restored.residual.copy()
            with self.assertRaises(ValueError):
                restored.load_state(path.with_name('missing.npz') if self.rank == 0 else path)
            np.testing.assert_array_equal(restored.residual, before)
            if self.rank == 0:
                np.savez(path, version=np.float64(np.inf), rank=np.int64(self.rank),
                         world_size=np.int64(self.comm.size), anchor=self.anchor,
                         residual=np.zeros_like(self.anchor), sample_factors=np.empty(0))
            with self.assertRaises(ValueError):
                restored.load_state(path)
            np.testing.assert_array_equal(restored.residual, before)

    def test_zero_update_initial_memory_and_uniform_fallback(self):
        for policy in ('uniform', 'sample_weighted'):
            transport = self.transport()
            result, _, _ = transport.synchronize(self.anchor.copy(), 0, policy)
            np.testing.assert_array_equal(result, self.anchor)
            np.testing.assert_array_equal(transport.residual, np.zeros_like(self.anchor))
        with self.assertRaises(ValueError):
            FP16DeltaEFCommunicator(self.comm, 'allreduce')


if __name__ == '__main__':
    unittest.main()
