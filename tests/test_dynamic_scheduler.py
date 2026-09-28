import csv
import json
import os
import tempfile
import unittest

import numpy as np

from src.metrics.v6_metrics import SCHEDULER_COLUMNS, V6MetricsRecorder
from src.training.dynamic_scheduler import DynamicWorkloadScheduler, PerformanceHistory
from src.training.trainer_joint import (
    _epoch_worker_indices,
    _gradient_weight,
    _scheduler_modes,
)


GPU = "lab01_gpu"
CPU = "lab01_cpu"
CANDIDATES = [
    {GPU: 224, CPU: 32},
    {GPU: 232, CPU: 24},
    {GPU: 234, CPU: 22},
    {GPU: 240, CPU: 16},
]


def make_scheduler(apply_changes=False, **overrides):
    kwargs = {
        "candidates": CANDIDATES,
        "global_batch": 256,
        "worker_device_types": {GPU: "gpu", CPU: "cpu"},
        "fallback_allocation": {GPU: 232, CPU: 24},
        "enabled": True,
        "apply_changes": apply_changes,
        "ema_alpha": 0.15,
        "decision_interval_steps": 1,
        "cooldown_steps": 3,
        "delta_threshold_ms": 5.0,
        "min_improvement_pct": 0.0,
        "warmup_steps": 0,
        "settling_steps": 2,
        "stability_windows": 1,
    }
    kwargs.update(overrides)
    return DynamicWorkloadScheduler(**kwargs)


class PerformanceHistoryTests(unittest.TestCase):
    def test_ema_and_invalid_observations(self):
        history = PerformanceHistory(ema_alpha=0.25)
        first = history.observe(GPU, "gpu", 232, 200.0)
        second = history.observe(GPU, "gpu", 232, 240.0)
        self.assertAlmostEqual(first.ema_time_ms, 200.0)
        self.assertAlmostEqual(second.ema_time_ms, 210.0)
        self.assertEqual(second.sample_count, 2)
        self.assertIsNone(history.observe(GPU, "gpu", 232, float("nan")))
        self.assertIsNone(history.observe(GPU, "gpu", 232, float("inf")))
        self.assertIsNone(history.observe(GPU, "gpu", 232, -1.0))
        self.assertEqual(history.profile_at(GPU, 232).sample_count, 2)

    def test_exact_profile_precedes_interpolation_and_throughput(self):
        history = PerformanceHistory(ema_alpha=1.0)
        history.observe(GPU, "gpu", 224, 210.0)
        history.observe(GPU, "gpu", 240, 230.0)
        self.assertAlmostEqual(history.predict_time_ms(GPU, 224), 210.0)
        self.assertAlmostEqual(history.predict_time_ms(GPU, 232), 220.0)
        self.assertAlmostEqual(history.predict_time_ms(GPU, 248), 248.0 / (240.0 / 230.0))


class SchedulerDecisionTests(unittest.TestCase):
    def test_observe_only_predicts_without_switching(self):
        scheduler = make_scheduler(apply_changes=False)
        scheduler.observe(GPU, 232, 217.0)
        scheduler.observe(CPU, 24, 229.0)
        decision = scheduler.maybe_decide(1, {GPU: 232, CPU: 24})
        self.assertEqual(decision.action, "OBSERVE_ONLY")
        self.assertEqual(decision.target_allocation, {GPU: 234, CPU: 22})
        self.assertEqual(scheduler.switch_count, 0)

    def test_dynamic_switch_respects_cooldown(self):
        scheduler = make_scheduler(apply_changes=True)
        scheduler.observe(GPU, 232, 217.0)
        scheduler.observe(CPU, 24, 229.0)
        first = scheduler.maybe_decide(1, {GPU: 232, CPU: 24})
        self.assertEqual(first.action, "SWITCH")
        self.assertEqual(first.target_allocation, {GPU: 234, CPU: 22})

        scheduler.observe(GPU, 234, 230.0)
        scheduler.observe(CPU, 22, 200.0)
        second = scheduler.maybe_decide(2, {GPU: 234, CPU: 22})
        self.assertEqual(second.action, "KEEP")
        self.assertEqual(second.reason, "cooldown_active")
        self.assertEqual(second.cooldown_remaining, 2)

    def test_post_switch_settling_observations_are_skipped(self):
        scheduler = make_scheduler(apply_changes=True, settling_steps=2)
        scheduler.observe(GPU, 232, 217.0)
        scheduler.observe(CPU, 24, 229.0)
        decision = scheduler.maybe_decide(1, {GPU: 232, CPU: 24})
        self.assertEqual(decision.action, "SWITCH")
        self.assertFalse(scheduler.should_observe(2))
        self.assertFalse(scheduler.should_observe(3))
        self.assertTrue(scheduler.should_observe(4))

    def test_target_must_be_stable_before_switch(self):
        scheduler = make_scheduler(
            apply_changes=True,
            cooldown_steps=0,
            stability_windows=2,
        )
        scheduler.observe(GPU, 232, 217.0)
        scheduler.observe(CPU, 24, 229.0)
        first = scheduler.maybe_decide(1, {GPU: 232, CPU: 24})
        second = scheduler.maybe_decide(2, {GPU: 232, CPU: 24})
        self.assertEqual(first.action, "KEEP")
        self.assertEqual(first.reason, "target_stability_1/2")
        self.assertEqual(second.action, "SWITCH")

    def test_minimum_improvement_prevents_small_switch(self):
        scheduler = make_scheduler(apply_changes=True, min_improvement_pct=10.0)
        scheduler.observe(GPU, 232, 217.0)
        scheduler.observe(CPU, 24, 229.0)
        decision = scheduler.maybe_decide(1, {GPU: 232, CPU: 24})
        self.assertEqual(decision.action, "KEEP")
        self.assertEqual(decision.reason, "improvement_below_threshold")

    def test_invalid_candidates_are_filtered_and_fallback_is_kept(self):
        scheduler = DynamicWorkloadScheduler(
            candidates=[{GPU: 200, CPU: 20}, {GPU: -1, CPU: 257}, {"bad": 256}],
            global_batch=256,
            worker_device_types={GPU: "gpu", CPU: "cpu"},
            fallback_allocation={GPU: 232, CPU: 24},
        )
        self.assertEqual(scheduler.candidates, [{GPU: 232, CPU: 24}])

    def test_three_worker_interface_uses_generic_critical_and_delta(self):
        third = "lab02_cpu"
        scheduler = DynamicWorkloadScheduler(
            candidates=[
                {GPU: 200, CPU: 28, third: 28},
                {GPU: 220, CPU: 18, third: 18},
            ],
            global_batch=256,
            worker_device_types={GPU: "gpu", CPU: "cpu", third: "cpu"},
            fallback_allocation={GPU: 200, CPU: 28, third: 28},
            delta_threshold_ms=5.0,
        )
        scheduler.observe(GPU, 200, 200.0)
        scheduler.observe(CPU, 28, 200.0)
        scheduler.observe(third, 28, 200.0)
        profiles = scheduler.history.latest_profiles([GPU, CPU, third])
        prediction = scheduler.solve(profiles, 256)
        self.assertEqual(set(prediction.allocation), {GPU, CPU, third})
        self.assertAlmostEqual(prediction.delta_ms, 0.0)


class IntegrationInvariantTests(unittest.TestCase):
    def test_disabled_and_observe_modes(self):
        self.assertEqual(_scheduler_modes({}), (False, False))
        self.assertEqual(
            _scheduler_modes({"dynamic_scheduler": {"enabled": False, "apply_changes": True}}),
            (False, False),
        )
        self.assertEqual(
            _scheduler_modes({"dynamic_scheduler": {"enabled": True, "apply_changes": False}}),
            (True, False),
        )

    def test_gradient_weighting_uses_actual_batch(self):
        gpu_gradient = np.array([2.0, -1.0], dtype=np.float32)
        cpu_gradient = np.array([6.0, 3.0], dtype=np.float32)
        for gpu_batch, cpu_batch in [(224, 32), (232, 24), (234, 22), (240, 16)]:
            self.assertEqual(gpu_batch + cpu_batch, 256)
            merged = (
                gpu_gradient * _gradient_weight(gpu_batch, 256)
                + cpu_gradient * _gradient_weight(cpu_batch, 256)
            )
            expected = (gpu_batch * gpu_gradient + cpu_batch * cpu_gradient) / 256.0
            np.testing.assert_allclose(merged, expected)

    def test_dynamic_partitions_are_disjoint_and_keep_global_batch(self):
        for gpu_batch, cpu_batch in [(224, 32), (232, 24), (240, 16)]:
            gpu_indices, num_steps = _epoch_worker_indices(
                1024, gpu_batch, cpu_batch, epoch=3, seed=42, worker="gpu"
            )
            cpu_indices, cpu_steps = _epoch_worker_indices(
                1024, gpu_batch, cpu_batch, epoch=3, seed=42, worker="cpu"
            )
            self.assertEqual(num_steps, 4)
            self.assertEqual(cpu_steps, num_steps)
            gpu_rows = gpu_indices.reshape(num_steps, gpu_batch)
            cpu_rows = cpu_indices.reshape(num_steps, cpu_batch)
            for step in range(num_steps):
                global_indices = np.concatenate([gpu_rows[step], cpu_rows[step]])
                self.assertEqual(len(global_indices), 256)
                self.assertEqual(len(set(global_indices.tolist())), 256)

    def test_scheduler_csv_and_summary(self):
        scheduler = make_scheduler(apply_changes=False)
        scheduler.observe(GPU, 232, 217.0)
        scheduler.observe(CPU, 24, 229.0)
        decision = scheduler.maybe_decide(1, {GPU: 232, CPU: 24})

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = V6MetricsRecorder(temp_dir, warmup_steps=0)
            recorder.record_decision(1, decision)
            recorder.record_step(1, 232, 24, 217.0, 229.0, 270.0)
            recorder.record_epoch(47.0, 950.0, 0.75)
            summary = recorder.finalize()

            with open(recorder.csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], SCHEDULER_COLUMNS)
            self.assertEqual(rows[1][18], "OBSERVE_ONLY")
            self.assertAlmostEqual(summary["mean_critical_time_ms"], 229.0)
            self.assertAlmostEqual(summary["mean_abs_gpu_cpu_delta_ms"], 12.0)
            self.assertEqual(summary["scheduler_decisions"], 1)
            self.assertEqual(summary["actual_switches"], 0)
            self.assertTrue(os.path.exists(recorder.summary_path))
            with open(recorder.summary_path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["final_validation_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
