import csv
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from analyze_runtime_stage_trace import analyze, split_collective_residence
from src.training.round_diagnostic import RoundDiagnostic


class RuntimeStageTests(unittest.TestCase):
    def test_residence_partition_conserves_interval(self):
        self.assertEqual(split_collective_residence(1, 5, 3), (2000, 2000))
        self.assertEqual(split_collective_residence(3, 5, 2), (0, 2000))
        self.assertEqual(split_collective_residence(1, 2, 3), (1000, 0))
        with self.assertRaises(ValueError):
            split_collective_residence(5, 1, 3)

    def test_buffered_stage_logging_does_not_require_mpi_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No Barrier/gather/send/recv methods: any added communication fails.
            comm = SimpleNamespace(Get_size=lambda: 1)
            diag = RoundDiagnostic(comm, 0, tmp, detailed=True)
            with diag.stage('checkpoint_last', 1):
                value = 42
            self.assertEqual(value, 42)
            self.assertFalse((Path(tmp) / 'runtime_stages_rank0.csv').exists())
            diag.flush()
            with (Path(tmp) / 'runtime_stages_rank0.csv').open() as stream:
                records = list(csv.DictReader(stream))
            self.assertEqual(len(records), 1)
            self.assertGreaterEqual(float(records[0]['elapsed_ms']), 0)

    def test_detailed_sync_preserves_weights_and_collective_sequence(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import numpy as np
        from src.training.trainer_mpi import MPITrainer

        class Comm:
            def __init__(self):
                self.calls = []

            def Allreduce(self, send, receive, op):
                self.calls.append(('Allreduce', op))
                receive[:] = send

            def allreduce(self, value, op):
                self.calls.append(('allreduce', op))
                return value

        class Model:
            def __init__(self):
                self.weights = [np.array([.25, -.75], dtype=np.float32)]

            def get_weights(self):
                return [w.copy() for w in self.weights]

            def set_weights(self, values):
                self.weights = [w.copy() for w in values]

        outputs = []
        for detailed in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                trainer = MPITrainer.__new__(MPITrainer)
                trainer.comm = Comm()
                trainer.model = Model()
                trainer.runtime_stage_trace_enabled = detailed
                trainer.delta_communicator = trainer.fp16_communicator = None
                trainer.rank = 0
                trainer.world_size = 1
                trainer.is_drift_adaptive = False
                trainer.logger = SimpleNamespace(run_dir=tmp, info=lambda *args: None)
                trainer.synchronize_model_weights(samples_since_sync=12000)
                outputs.append((trainer.model.weights, trainer.comm.calls))
                if detailed:
                    timing = trainer.last_sync_detail
                    self.assertLessEqual(timing['collective_enter'], timing['collective_exit'])
                    self.assertLessEqual(timing['collective_exit'], timing['sample_scalar_enter'])
                    self.assertLessEqual(timing['set_weights_exit'], timing['core_sync_exit'])
        np.testing.assert_array_equal(outputs[0][0][0], outputs[1][0][0])
        self.assertEqual(outputs[0][1], outputs[1][1])
        self.assertEqual(len(outputs[1][1]), 4)

    def test_one_rank_cpu_training_emits_complete_stage_files(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import numpy as np
        import tensorflow as tf
        from mpi4py import MPI
        from src.metrics.logger import ExperimentLogger
        from src.training.trainer_mpi import MPITrainer
        with tempfile.TemporaryDirectory() as tmp:
            tf.random.set_seed(42)
            model = tf.keras.Sequential([tf.keras.layers.Input(shape=(4,)), tf.keras.layers.Dense(3)])
            data = tf.data.Dataset.from_tensor_slices((np.ones((4, 4), dtype=np.float32),
                                                     np.array([0, 1, 2, 0], dtype=np.int32))).batch(2)
            config = dict(experiment=dict(name='tiny_runtime_trace'), mode='mpi', seed=42,
                training=dict(epochs=2, global_batch_size=2, rank_batch_sizes=[2],
                    optimizer='adamw', learning_rate=.001, sync_mode='local_sgd', local_sgd_h=1,
                    comm_precision='fp32', avg_policy='sample_weighted', optimizer_state_sync='preserve_local'),
                logging=dict(output_dir=tmp, round_diagnostic=True, runtime_stage_trace=True, log_interval=99))
            logger = ExperimentLogger(config)
            trainer = MPITrainer(MPI.COMM_WORLD, 0, 1, None, model, config, data, data, data, 2, 1, logger)
            trainer.train()
            folder = Path(logger.run_dir)
            with (folder / 'sync_trace_rank0.csv').open() as stream:
                syncs = list(csv.DictReader(stream))
            with (folder / 'runtime_stages_rank0.csv').open() as stream:
                stages = list(csv.DictReader(stream))
            self.assertEqual(len(syncs), 4)
            self.assertEqual(sum(r['stage'] == 'iterator_create' for r in stages), 2)
            self.assertEqual(sum(r['stage'] == 'checkpoint_last' for r in stages), 2)
            self.assertEqual(sum(r['stage'] == 'clock_calibration_and_flush' for r in stages), 2)
            destination = Path(tmp) / 'analysis_input/ranks/gpu_only/rank0'
            destination.parent.mkdir(parents=True)
            destination.symlink_to(folder, target_is_directory=True)
            report = analyze(Path(tmp) / 'analysis_input')
            self.assertEqual(report['variants']['gpu_only']['ranks'][0]['rounds'], 4)


if __name__ == '__main__':
    unittest.main()
