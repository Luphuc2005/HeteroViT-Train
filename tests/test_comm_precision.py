import unittest
import numpy as np
from mpi4py import MPI
from src.training.weight_communication import average_weights


class TestWeightCommunication(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.N = 8192
        np.random.seed(42 + self.rank)
        self.weights = np.random.randn(self.N).astype(np.float32) * 0.1

    def test_fp32_allreduce(self):
        samples = 100 * (self.rank + 1)
        res, total, stats = average_weights(
            self.comm, self.weights, samples, policy="sample_weighted", precision="fp32"
        )
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.isfinite(res).all())
        self.assertEqual(stats["communication_dtype"], "fp32")
        self.assertEqual(stats["collective"], "allreduce")
        self.assertEqual(stats["payload_bytes"], self.N * 4)
        self.assertEqual(stats["num_elements"], self.N)
        self.assertGreater(stats["communication_ms"], 0.0)

    def test_fp16_allgather(self):
        samples = 100 * (self.rank + 1)
        res, total, stats = average_weights(
            self.comm, self.weights, samples, policy="sample_weighted", precision="fp16", collective="allgather"
        )
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.isfinite(res).all())
        self.assertEqual(stats["communication_dtype"], "fp16")
        self.assertEqual(stats["collective"], "allgather")
        self.assertEqual(stats["payload_bytes"], self.N * 2)
        self.assertEqual(stats["num_elements"], self.N)
        self.assertGreaterEqual(stats["cast_to_fp16_ms"], 0.0)
        self.assertGreaterEqual(stats["communication_ms"], 0.0)
        self.assertGreaterEqual(stats["cast_accumulation_ms"], 0.0)

    def test_fp16_allreduce(self):
        samples = 100 * (self.rank + 1)
        res, total, stats = average_weights(
            self.comm, self.weights, samples, policy="sample_weighted", precision="fp16", collective="allreduce"
        )
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.isfinite(res).all())
        self.assertEqual(stats["communication_dtype"], "fp16")
        self.assertEqual(stats["collective"], "allreduce")
        self.assertEqual(stats["payload_bytes"], self.N * 2)
        self.assertEqual(stats["num_elements"], self.N)
        self.assertGreaterEqual(stats["cast_to_fp16_ms"], 0.0)
        self.assertGreaterEqual(stats["communication_ms"], 0.0)
        self.assertGreaterEqual(stats["cast_accumulation_ms"], 0.0)

    def test_fp16_error_bounds_against_fp32(self):
        # With identical weights across ranks, FP16 should match FP32 within FP16 precision
        identical_w = np.sin(np.linspace(0, 10, self.N)).astype(np.float32)
        res_fp32, _, _ = average_weights(
            self.comm, identical_w, 100, policy="uniform", precision="fp32"
        )
        res_fp16_ag, _, _ = average_weights(
            self.comm, identical_w, 100, policy="uniform", precision="fp16", collective="allgather"
        )
        res_fp16_ar, _, _ = average_weights(
            self.comm, identical_w, 100, policy="uniform", precision="fp16", collective="allreduce"
        )
        # Difference between FP32 and FP16 recovered should be within 1e-3
        max_err_ag = np.max(np.abs(res_fp32 - res_fp16_ag))
        max_err_ar = np.max(np.abs(res_fp32 - res_fp16_ar))
        self.assertLess(max_err_ag, 1e-3)
        self.assertLess(max_err_ar, 1e-3)


if __name__ == "__main__":
    unittest.main()

