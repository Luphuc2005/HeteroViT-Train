"""Runtime Telemetry and Step Instrumentation for Distributed Training.

Logs exact step breakdown per node without modifying training semantics:
- step_id, node_id, rank, local_batch
- compute_start, compute_end -> compute_ms
- sync_enter, sync_exit -> sync_wait_ms
- comm_start, comm_end -> comm_ms
- step_wall_time -> step_ms
Maintains running straggler statistics: runtime_ema, runtime_mean, runtime_std.
"""
import os
import csv
import json
import math
from typing import Dict, Any, List, Optional
import numpy as np


STEP_METRICS_HEADER = [
    "step_id",
    "node_id",
    "rank",
    "local_batch",
    "compute_start",
    "compute_end",
    "sync_enter",
    "sync_exit",
    "comm_start",
    "comm_end",
    "step_wall_time",
    "compute_ms",
    "sync_wait_ms",
    "comm_ms",
    "step_ms",
]


class RuntimeProfiler:
    """Step instrumentation collector and straggler statistics maintainer."""

    def __init__(
        self,
        output_dir: str,
        node_id: str,
        rank: int,
        ema_alpha: float = 0.15,
        filename: str = "step_metrics.csv",
    ):
        self.output_dir = output_dir
        self.node_id = node_id
        self.rank = rank
        self.ema_alpha = float(ema_alpha)
        self.csv_path = os.path.join(output_dir, filename)

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(STEP_METRICS_HEADER)

        self.step_times_ms: List[float] = []
        self.compute_times_ms: List[float] = []
        self.sync_wait_times_ms: List[float] = []
        self.comm_times_ms: List[float] = []

        self.runtime_ema: Optional[float] = None
        self.compute_ema: Optional[float] = None

    def record_step(
        self,
        step_id: int,
        local_batch: int,
        compute_start: float,
        compute_end: float,
        sync_enter: float,
        sync_exit: float,
        comm_start: float,
        comm_end: float,
        step_wall_time: float,
    ) -> Dict[str, float]:
        """Records timestamps, calculates durations, appends to CSV and updates stats."""
        compute_ms = max(0.0, (compute_end - compute_start) * 1000.0)
        sync_wait_ms = max(0.0, (sync_exit - sync_enter) * 1000.0)
        comm_ms = max(0.0, (comm_end - comm_start) * 1000.0)
        step_ms = max(0.0, step_wall_time * 1000.0)

        # Update running EMAs
        if self.runtime_ema is None:
            self.runtime_ema = step_ms
        else:
            self.runtime_ema = self.ema_alpha * step_ms + (1.0 - self.ema_alpha) * self.runtime_ema

        if self.compute_ema is None:
            self.compute_ema = compute_ms
        else:
            self.compute_ema = self.ema_alpha * compute_ms + (1.0 - self.ema_alpha) * self.compute_ema

        self.step_times_ms.append(step_ms)
        self.compute_times_ms.append(compute_ms)
        self.sync_wait_times_ms.append(sync_wait_ms)
        self.comm_times_ms.append(comm_ms)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step_id,
                self.node_id,
                self.rank,
                local_batch,
                f"{compute_start:.6f}",
                f"{compute_end:.6f}",
                f"{sync_enter:.6f}",
                f"{sync_exit:.6f}",
                f"{comm_start:.6f}",
                f"{comm_end:.6f}",
                f"{step_wall_time:.6f}",
                f"{compute_ms:.2f}",
                f"{sync_wait_ms:.2f}",
                f"{comm_ms:.2f}",
                f"{step_ms:.2f}",
            ])

        return {
            "compute_ms": compute_ms,
            "sync_wait_ms": sync_wait_ms,
            "comm_ms": comm_ms,
            "step_ms": step_ms,
            "runtime_ema": self.runtime_ema,
            "compute_ema": self.compute_ema,
        }

    def get_latest_telemetry(self, current_batch: int = 0) -> Dict[str, Any]:
        """Returns the most recent step timing breakdown and EMA stats."""
        latest_compute = self.compute_times_ms[-1] if self.compute_times_ms else 0.0
        latest_comm = self.comm_times_ms[-1] if self.comm_times_ms else 0.0
        latest_sync = self.sync_wait_times_ms[-1] if self.sync_wait_times_ms else 0.0
        valid_compute = self.compute_times_ms[-20:] if len(self.compute_times_ms) >= 2 else self.compute_times_ms
        compute_std = float(np.std(valid_compute)) if valid_compute else 0.0

        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "current_batch": current_batch,
            "compute_ms": latest_compute,
            "compute_ema_ms": self.compute_ema if self.compute_ema is not None else latest_compute,
            "runtime_std_ms": compute_std,
            "comm_ms": latest_comm,
            "idle_ms": latest_sync,
            "active": current_batch > 0,
        }

    def get_summary_statistics(self, warmup_discard: int = 5) -> Dict[str, Any]:
        """Calculates runtime_mean, runtime_std, runtime_ema, excluding initial warmup."""
        valid_steps = self.step_times_ms[warmup_discard:] if len(self.step_times_ms) > warmup_discard else self.step_times_ms
        valid_compute = self.compute_times_ms[warmup_discard:] if len(self.compute_times_ms) > warmup_discard else self.compute_times_ms
        valid_wait = self.sync_wait_times_ms[warmup_discard:] if len(self.sync_wait_times_ms) > warmup_discard else self.sync_wait_times_ms
        valid_comm = self.comm_times_ms[warmup_discard:] if len(self.comm_times_ms) > warmup_discard else self.comm_times_ms

        if not valid_steps:
            return {
                "node_id": self.node_id,
                "rank": self.rank,
                "sample_count": 0,
                "runtime_mean_ms": 0.0,
                "runtime_std_ms": 0.0,
                "runtime_ema_ms": 0.0,
                "compute_mean_ms": 0.0,
                "sync_wait_mean_ms": 0.0,
                "comm_mean_ms": 0.0,
            }

        return {
            "node_id": self.node_id,
            "rank": self.rank,
            "sample_count": len(valid_steps),
            "runtime_mean_ms": float(np.mean(valid_steps)),
            "runtime_std_ms": float(np.std(valid_steps)),
            "runtime_ema_ms": float(self.runtime_ema) if self.runtime_ema is not None else float(np.mean(valid_steps)),
            "compute_mean_ms": float(np.mean(valid_compute)),
            "sync_wait_mean_ms": float(np.mean(valid_wait)),
            "comm_mean_ms": float(np.mean(valid_comm)),
        }
