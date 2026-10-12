import unittest
import numpy as np
from mpi4py import MPI
from src.training.fp16_delta_communicator import FP16DeltaCommunicator, DELTA_MODES
from src.training.weight_communication import average_weights


class TestDelta(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.anchor = np.full(128, 100.123, dtype=np.float32)

    def test_rounds_and_weighted_reference(self):
        for mode in DELTA_MODES:
            transport = FP16DeltaCommunicator(self.comm, mode[len('fp16_delta_'):])
            transport.set_anchor(self.anchor)
            original_anchor = self.anchor.copy()
            for policy, samples in [('sample_weighted', self.rank + 1), ('uniform', 1), ('sample_weighted', 0)]:
                previous = transport.global_prev.copy()
                local = previous + np.float32((self.rank + 1) * 0.01)
                reference, _, _ = average_weights(self.comm, local, samples, policy)
                result, _, stats = transport.synchronize(local, samples, policy)
                np.testing.assert_allclose(result, reference, rtol=0, atol=1e-4)
                np.testing.assert_array_equal(self.anchor, original_anchor)
                if mode.endswith('allgather'):
                    rows = self.comm.allgather((local - previous).astype(np.float16))
                    counts = self.comm.allgather(float(samples))
                    factors = counts if policy == 'sample_weighted' and sum(counts) > 0 else [1.] * len(rows)
                    expected = np.zeros_like(previous)
                    for row, count in zip(rows, factors):
                        expected += row.astype(np.float32) * np.float32(count)
                    expected /= np.float32(sum(factors))
                    np.testing.assert_array_equal(result, previous + expected)
                self.assertEqual(result.dtype, np.float32)
                self.assertTrue(np.isfinite(result).all())
                self.assertEqual(stats['method'], mode)
                self.assertEqual(stats['payload_bytes'], local.size * 2)
                for key in ('encode_ms', 'communication_ms', 'decode_ms', 'sync_total_ms',
                            'relative_l2_quantization_error', 'delta_l2_norm'):
                    self.assertTrue(np.isfinite(stats[key]))
                    self.assertGreaterEqual(stats[key], 0)
                for row in self.comm.allgather(result):
                    np.testing.assert_array_equal(row, result)
                np.testing.assert_array_equal(local, previous + np.float32((self.rank + 1) * 0.01))
                zero_result, _, stats = transport.synchronize(result.copy(), samples, policy)
                np.testing.assert_array_equal(zero_result, result)
                self.assertEqual(stats['delta_l2_norm'], 0)
                self.assertEqual(stats['relative_l2_quantization_error'], 0)

    def test_zero_preserves_nonrepresentable_anchor(self):
        for mode in DELTA_MODES:
            result, _, _ = average_weights(self.comm, self.anchor, 1, precision=mode, global_prev=self.anchor)
            np.testing.assert_array_equal(result, self.anchor)

    def test_anchor_mismatch(self):
        if self.comm.Get_size() < 2:
            self.skipTest('Requires multiple ranks')
        with self.assertRaises(ValueError):
            FP16DeltaCommunicator(self.comm).set_anchor(self.anchor + np.float32(self.rank))

    def test_collective_rejection(self):
        for collective in ('allgather', 'allreduce'):
            transport = FP16DeltaCommunicator(self.comm, collective)
            transport.set_anchor(self.anchor)
            for value in (np.nan, np.inf, 1e10):
                local = self.anchor.copy()
                if self.rank == 0:
                    local[0] = value
                with self.assertRaises((ValueError, FloatingPointError)):
                    transport.synchronize(local, 1)
                np.testing.assert_array_equal(transport.global_prev, self.anchor)


if __name__ == '__main__':
    unittest.main()
