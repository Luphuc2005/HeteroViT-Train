"""CSV decision logging and end-of-run summary for the V6 scheduler."""
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Mapping

import numpy as np

from src.training.dynamic_scheduler import SchedulerDecision


SCHEDULER_COLUMNS = [
    "step",
    "epoch",
    "current_gpu_batch",
    "current_cpu_batch",
    "gpu_time_raw_ms",
    "cpu_time_raw_ms",
    "gpu_time_ema_ms",
    "cpu_time_ema_ms",
    "gpu_throughput",
    "cpu_throughput",
    "delta_ms",
    "critical_ms",
    "predicted_gpu_batch",
    "predicted_cpu_batch",
    "predicted_gpu_time_ms",
    "predicted_cpu_time_ms",
    "predicted_delta_ms",
    "predicted_critical_ms",
    "action",
    "reason",
    "cooldown_remaining",
]


class V6MetricsRecorder:
    """Collects V6 step statistics without changing the existing training logs."""

    def __init__(
        self,
        results_dir: str,
        warmup_steps: int,
        gpu_worker_id: str = "lab01_gpu",
        cpu_worker_id: str = "lab01_cpu",
    ):
        self.results_dir = results_dir
        self.warmup_steps = max(0, int(warmup_steps))
        self.gpu_worker_id = gpu_worker_id
        self.cpu_worker_id = cpu_worker_id
        self.csv_path = os.path.join(results_dir, "v6_scheduler.csv")
        self.summary_path = os.path.join(results_dir, "v6_summary.json")
        self.gpu_times: List[float] = []
        self.cpu_times: List[float] = []
        self.idle_gaps: List[float] = []
        self.critical_times: List[float] = []
        self.epoch_times: List[float] = []
        self.epoch_throughputs: List[float] = []
        self.val_accuracies: List[float] = []
        self.split_wall_ms: Dict[str, float] = defaultdict(float)
        self.split_steps: Dict[str, int] = defaultdict(int)
        self.decision_count = 0
        self.switch_count = 0

        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(SCHEDULER_COLUMNS)

    @staticmethod
    def _valid_timing(value: float) -> bool:
        return math.isfinite(value) and value > 0.0

    def record_step(
        self,
        global_step: int,
        gpu_batch: int,
        cpu_batch: int,
        gpu_time_ms: float,
        cpu_time_ms: float,
        step_wall_ms: float,
    ) -> None:
        split_key = f"{int(gpu_batch)}/{int(cpu_batch)}"
        if self._valid_timing(step_wall_ms):
            self.split_wall_ms[split_key] += float(step_wall_ms)
        self.split_steps[split_key] += 1

        if int(global_step) <= self.warmup_steps:
            return
        if not self._valid_timing(gpu_time_ms) or not self._valid_timing(cpu_time_ms):
            return
        self.gpu_times.append(float(gpu_time_ms))
        self.cpu_times.append(float(cpu_time_ms))
        self.idle_gaps.append(abs(float(gpu_time_ms) - float(cpu_time_ms)))
        self.critical_times.append(max(float(gpu_time_ms), float(cpu_time_ms)))

    def record_decision(self, epoch: int, decision: SchedulerDecision) -> None:
        gpu_profile = decision.profiles[self.gpu_worker_id]
        cpu_profile = decision.profiles[self.cpu_worker_id]
        current = decision.current_prediction
        target = decision.target_prediction
        row = [
            decision.step,
            int(epoch),
            decision.current_allocation[self.gpu_worker_id],
            decision.current_allocation[self.cpu_worker_id],
            f"{gpu_profile.raw_time_ms:.3f}",
            f"{cpu_profile.raw_time_ms:.3f}",
            f"{gpu_profile.ema_time_ms:.3f}",
            f"{cpu_profile.ema_time_ms:.3f}",
            f"{gpu_profile.throughput_img_per_ms:.6f}",
            f"{cpu_profile.throughput_img_per_ms:.6f}",
            f"{current.delta_ms:.3f}",
            f"{current.critical_ms:.3f}",
            decision.target_allocation[self.gpu_worker_id],
            decision.target_allocation[self.cpu_worker_id],
            f"{target.worker_times_ms[self.gpu_worker_id]:.3f}",
            f"{target.worker_times_ms[self.cpu_worker_id]:.3f}",
            f"{target.delta_ms:.3f}",
            f"{target.critical_ms:.3f}",
            decision.action,
            decision.reason,
            decision.cooldown_remaining,
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        self.decision_count += 1
        if decision.action == "SWITCH":
            self.switch_count += 1

    def record_epoch(self, epoch_time: float, throughput: float, val_accuracy: float) -> None:
        self.epoch_times.append(float(epoch_time))
        self.epoch_throughputs.append(float(throughput))
        self.val_accuracies.append(float(val_accuracy))

    @staticmethod
    def _mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        return float(np.percentile(values, percentile)) if values else 0.0

    def finalize(self) -> Dict[str, Any]:
        total_wall = sum(self.split_wall_ms.values())
        total_steps = sum(self.split_steps.values())
        split_percentages: Dict[str, Dict[str, float]] = {}
        for split in sorted(self.split_steps):
            split_percentages[split] = {
                "wall_time_pct": (
                    self.split_wall_ms[split] / total_wall * 100.0 if total_wall > 0.0 else 0.0
                ),
                "step_pct": (
                    self.split_steps[split] / total_steps * 100.0 if total_steps > 0 else 0.0
                ),
                "steps": self.split_steps[split],
            }

        summary: Dict[str, Any] = {
            "measurement_scope": f"valid steps after global warmup step {self.warmup_steps}",
            "avg_epoch_time_s": self._mean(self.epoch_times),
            "avg_throughput_img_s": self._mean(self.epoch_throughputs),
            "mean_gpu_time_ms": self._mean(self.gpu_times),
            "mean_cpu_time_ms": self._mean(self.cpu_times),
            "mean_idle_gap_ms": self._mean(self.idle_gaps),
            "p50_idle_gap_ms": self._percentile(self.idle_gaps, 50.0),
            "p95_idle_gap_ms": self._percentile(self.idle_gaps, 95.0),
            "max_idle_gap_ms": max(self.idle_gaps) if self.idle_gaps else 0.0,
            "mean_critical_time_ms": self._mean(self.critical_times),
            "mean_abs_gpu_cpu_delta_ms": self._mean(self.idle_gaps),
            "scheduler_decisions": self.decision_count,
            "actual_switches": self.switch_count,
            "split_time_percentages": split_percentages,
            "best_validation_accuracy": max(self.val_accuracies) if self.val_accuracies else 0.0,
            "final_validation_accuracy": self.val_accuracies[-1] if self.val_accuracies else 0.0,
        }
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        return summary


def format_v6_summary(summary: Mapping[str, Any]) -> str:
    splits = summary.get("split_time_percentages", {})
    split_text = ", ".join(
        f"{name}={values['wall_time_pct']:.1f}%"
        for name, values in splits.items()
    ) or "N/A"
    return (
        "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  >>> V6 DYNAMIC SCHEDULER SUMMARY\n"
        f"      Avg epoch: {summary['avg_epoch_time_s']:.2f}s | "
        f"Throughput: {summary['avg_throughput_img_s']:.1f} img/s\n"
        f"      Mean GPU/CPU: {summary['mean_gpu_time_ms']:.2f}/"
        f"{summary['mean_cpu_time_ms']:.2f} ms\n"
        f"      Mean critical: {summary['mean_critical_time_ms']:.2f} ms | "
        f"Mean |GPU-CPU|: {summary['mean_abs_gpu_cpu_delta_ms']:.2f} ms\n"
        f"      Idle gap p50/p95/max: {summary['p50_idle_gap_ms']:.2f}/"
        f"{summary['p95_idle_gap_ms']:.2f}/{summary['max_idle_gap_ms']:.2f} ms\n"
        f"      Decisions/switches: {summary['scheduler_decisions']}/"
        f"{summary['actual_switches']} | Split wall time: {split_text}\n"
        f"      Best/final val acc: {summary['best_validation_accuracy'] * 100.0:.2f}%/"
        f"{summary['final_validation_accuracy'] * 100.0:.2f}%\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
