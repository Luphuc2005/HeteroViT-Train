import unittest
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

from src.training.fp16_communicator import FP16Communicator
from src.training.fp16_diagnostics import SameInputFP16Diagnostic, install_same_input_diagnostic


class TestSameInputFP16Diagnostic(unittest.TestCase):
    def test_trainer_hook_applies_only_fp32_and_writes_shadow_report(self):
        comm = MPI.COMM_WORLD
        transport = FP16Communicator(comm, 'allgather')
        initial = np.full(128, 0.12345, dtype=np.float32)
        model = SimpleNamespace(weights=initial.copy())
        model.get_weights = lambda: [model.weights.copy()]

        def sync(policy, samples_since_sync, verify_weights):
            reference, _, _ = transport._compute_fp32_reference(
                model.weights, samples_since_sync, policy)
            model.weights = reference
            return ('original_sync_result',)

        with tempfile.TemporaryDirectory() as directory:
            trainer = SimpleNamespace(comm=comm, rank=comm.rank, comm_precision='fp32',
                max_steps=120, sync_mode='local_sgd', model=model,
                logger=SimpleNamespace(run_dir=directory), synchronize_model_weights=sync)
            install_same_input_diagnostic(trainer)
            model.weights = initial + np.float32((comm.rank + 1) * 1e-5)
            expected, _, _ = transport._compute_fp32_reference(
                model.weights.copy(), comm.rank + 1, 'sample_weighted')
            result = trainer.synchronize_model_weights(samples_since_sync=comm.rank + 1)
            self.assertEqual(result, ('original_sync_result',))
            np.testing.assert_array_equal(model.weights, expected)
            np.testing.assert_array_equal(trainer.fp16_shadow.delta.global_prev, expected)
            path = Path(directory) / ('fp16_same_inputs_rank%d.csv' % comm.rank)
            with path.open() as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual(len(records), 1)
            self.assertLess(float(records[0]['delta_error_l2']), float(records[0]['whole_error_l2']))

    def test_shadow_results_and_reference_anchor_across_rounds(self):
        comm = MPI.COMM_WORLD
        reference_transport = FP16Communicator(comm, 'allgather')
        anchor = np.linspace(0.12345, 1.12345, 256, dtype=np.float32)
        diagnostic = SameInputFP16Diagnostic(comm, anchor)
        for step in range(2):
            previous = anchor.copy()
            local = anchor + np.float32((comm.rank + 1) * 1e-5)
            before = local.copy()
            reference, _, _ = reference_transport._compute_fp32_reference(
                local, comm.rank + 1, 'sample_weighted')
            row = diagnostic.compare(local, reference, comm.rank + 1)
            counts = comm.allgather(comm.rank + 1)
            whole_rows = comm.allgather(local.astype(np.float16))
            delta_rows = comm.allgather((local - previous).astype(np.float16))
            whole = np.zeros_like(anchor)
            delta = np.zeros_like(anchor)
            for count, w, d in zip(counts, whole_rows, delta_rows):
                whole += w.astype(np.float32) * np.float32(count)
                delta += d.astype(np.float32) * np.float32(count)
            whole /= np.float32(sum(counts))
            delta = previous + delta / np.float32(sum(counts))
            for name, result in [('whole', whole), ('delta', delta)]:
                error = result.astype(np.float64) - reference.astype(np.float64)
                self.assertEqual(row[name + '_error_l2'], float(np.linalg.norm(error)))
                self.assertEqual(row[name + '_payload_bytes'], local.size * 2)
            self.assertLess(row['delta_error_l2'], row['whole_error_l2'])
            self.assertEqual(row['round'], step + 1)
            np.testing.assert_array_equal(local, before)
            np.testing.assert_array_equal(diagnostic.delta.global_prev, reference)
            # The shadows differ from the reference in this fixture; the next
            # anchor must track the actual FP32 model, not the shadow result.
            self.assertFalse(np.array_equal(whole, reference))
            anchor = reference.copy()

    def test_zero_update_reports_absolute_error_and_finite_relative_error(self):
        comm = MPI.COMM_WORLD
        anchor = np.full(32, 0.12345, dtype=np.float32)
        diagnostic = SameInputFP16Diagnostic(comm, anchor)
        # Exact common reference avoids attributing FP32 reduction roundoff to
        # a zero update. Test both uniform and zero-count fallback protocols.
        for policy in ('uniform', 'sample_weighted'):
            row = diagnostic.compare(anchor.copy(), anchor.copy(), 0, policy)
            self.assertEqual(row['reference_update_l2'], 0)
            self.assertEqual(row['delta_error_l2'], 0)
            self.assertGreater(row['whole_error_l2'], 0)
            self.assertTrue(np.isfinite(row['whole_error_relative_to_update']))


if __name__ == '__main__':
    unittest.main()
