"""Timing regressions for the baseline's duplicate/mismatched CSV schema."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('sweep_summary', Path(__file__).resolve().parents[1] / 'tools/summarize_fp16_sweep.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HEADER = ['communication_dtype', 'collective', 'num_elements', 'payload_bytes',
          'cast_to_fp16_ms', 'communication_ms', 'cast_accumulation_ms']


class TestSweepSummary(unittest.TestCase):
    def test_legacy_rows_are_counted_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'communication.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                for value in (10, 20):
                    row = ['fp16', 'allgather', 2, 4, 1, value, 2]
                    writer.writerow(row)
                    writer.writerow(row + [value + 5])
            rows = module.communication_rows(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(sum(float(r['communication_ms']) for r in rows), 30)
            self.assertEqual(sum(float(r['total_sync_ms']) for r in rows), 40)

    def test_maximum_of_rank_totals_and_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for rank in range(5):
                path = root / 'ranks/fp32' / ('rank%d' % rank) / ('weight_communication_rank%d.csv' % rank)
                path.parent.mkdir(parents=True)
                with path.open('w', newline='') as handle:
                    writer = csv.writer(handle)
                    writer.writerow(HEADER + ['total_sync_ms'])
                    for value in (rank + 1, rank + 2):
                        writer.writerow(['fp32', 'allreduce', 2, 8, 0, value, 0, value + 1])
            (root / 'fp32.log').write_text('MPI Local SGD Training finished in 123.45s\nFinal Test Evaluation -> Loss: 0.1, Accuracy: 87.65%\n')
            module.summarize(root)
            result = json.loads((root / 'summary.json').read_text())[0]
            self.assertEqual(result['synchronization_rounds'], 2)
            self.assertEqual(result['max_rank_communication_total_ms'], 11)
            self.assertEqual(result['max_rank_sync_total_ms'], 13)
            self.assertEqual(result['test_accuracy_percent'], 87.65)

    def test_malformed_schema_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'communication.csv'
            path.write_text(','.join(HEADER) + '\nfp16,allgather,2\n')
            with self.assertRaises(ValueError):
                module.communication_rows(path)


if __name__ == '__main__':
    unittest.main()
