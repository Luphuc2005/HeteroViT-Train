import unittest
import numpy as np
from mpi4py import MPI
from src.training.fp16_communicator import FP16Communicator


class TestFP16Communicator(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.N = 4096
        np.random.seed(123 + self.rank)
        self.weights = np.random.randn(self.N).astype(np.float32) * 0.05

    def test_invalid_collective(self):
        with self.assertRaises(ValueError):
            FP16Communicator(self.comm, collective="invalid_collective")

    def test_quantize_and_dequantize(self):
        comm = FP16Communicator(self.comm, collective="allreduce")
        q, t_down = comm.quantize(self.weights, factor=0.5)
        self.assertEqual(q.dtype, np.float16)
        self.assertEqual(q.size, self.weights.size)
        self.assertGreaterEqual(t_down, 0.0)

        recovered, t_up = comm.dequantize(q)
        self.assertEqual(recovered.dtype, np.float32)
        self.assertGreaterEqual(t_up, 0.0)
        # Check that recovered matches unscaled scaled weights within float16 epsilon
        np.testing.assert_allclose(recovered, self.weights * 0.5, rtol=1e-3, atol=1e-3)

    def test_synchronize_allreduce(self):
        comm = FP16Communicator(self.comm, collective="allreduce", verify_reference=True)
        samples = 50 * (self.rank + 1)
        res, total, stats = comm.synchronize(self.weights, samples, policy="sample_weighted")
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.isfinite(res).all())
        self.assertEqual(stats["collective"], "allreduce")
        self.assertEqual(stats["communication_dtype"], "fp16")
        self.assertIn("relative_l2_error", stats)
        self.assertLess(stats["relative_l2_error"], 0.01)

    def test_synchronize_allgather(self):
        comm = FP16Communicator(self.comm, collective="allgather", verify_reference=True)
        samples = 50 * (self.rank + 1)
        res, total, stats = comm.synchronize(self.weights, samples, policy="sample_weighted")
        self.assertEqual(res.dtype, np.float32)
        self.assertTrue(np.isfinite(res).all())
        self.assertEqual(stats["collective"], "allgather")
        self.assertEqual(stats["communication_dtype"], "fp16")
        self.assertIn("relative_l2_error", stats)
        self.assertLess(stats["relative_l2_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
