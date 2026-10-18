"""Tests for clock alignment and unchanged optimizer updates under telemetry."""
import sys
import unittest
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from paired_workload_analysis import align, stats


class ClockAlignmentTests(unittest.TestCase):
    def test_offset_and_linear_drift_are_removed(self):
        markers = [dict(local_timestamp=100.0, root_minus_local_seconds=-90.0, uncertainty_seconds=0.001),
                   dict(local_timestamp=200.0, root_minus_local_seconds=-89.9, uncertainty_seconds=0.003)]
        aligned, uncertainty = align(150.0, markers)
        self.assertAlmostEqual(aligned, 60.05)
        self.assertAlmostEqual(uncertainty, 0.002)

    def test_percentile_interpolation(self):
        self.assertAlmostEqual(stats([0, 10, 20])['p95'], 19.0)


class OptimizerTelemetryTests(unittest.TestCase):
    def test_cpu_timer_keeps_adamw_updates_and_moments(self):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        import numpy as np
        import tensorflow as tf
        from src.training.trainer_mpi import MPITrainer
        from src.training.base_trainer import get_optimizer
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.random.set_seed(42)
        trainers = []
        for enabled in [False, True]:
            trainer = MPITrainer.__new__(MPITrainer)
            trainer.rank = 1
            trainer.strategy = None
            trainer.local_batch_size = 6
            trainer.accum_steps = 1
            trainer.round_diagnostic_enabled = enabled
            trainer.model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(4,)),
                                                tf.keras.layers.Dense(3)])
            trainer.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            trainer.optimizer = get_optimizer(dict(optimizer='adamw', learning_rate=0.001, weight_decay=0.0001))
            trainer._setup_step_functions()
            trainers.append(trainer)
        trainers[1].model.set_weights(trainers[0].model.get_weights())
        images = tf.reshape(tf.range(24, dtype=tf.float32), (6, 4)) / 24
        labels = tf.constant([0, 1, 2, 0, 1, 2])
        for _ in range(4):
            base = trainers[0]._local_train_step(images, labels)
            timed = trainers[1]._local_train_step(images, labels)
            self.assertEqual(len(timed), 3)
            self.assertGreater(float(timed[2]), 0.0)
            np.testing.assert_allclose(float(base[0]), float(timed[0]), rtol=1e-6)
        for left, right in zip(trainers[0].model.get_weights(), trainers[1].model.get_weights()):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        for left, right in zip(trainers[0].optimizer.variables(), trainers[1].optimizer.variables()):
            np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-6, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
