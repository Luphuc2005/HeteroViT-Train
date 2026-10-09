"""Structured CSV and JSON Logging for Adaptive Local SGD.

Generates machine-readable audit artifacts:
  - sync_events.csv: Complete telemetry of each synchronization round.
  - controller_decisions.csv: Decision evaluations, candidate rankings, and budgets.
  - epoch_metrics.csv: Epoch-level training/validation accuracy, loss, and times.
  - runtime_breakdown.csv: Cumulative compute, sync, validation, and overhead times.
  - run_metadata.json: Complete frozen configuration and system environment snapshot.
"""

import os
import csv
import json
import time
from typing import Dict, Any, Optional
from .controller_base import SyncRecord, AdaptiveDecision


class AdaptiveStructuredLogger:
    """Manages structured machine-readable logging for adaptive training runs."""

    def __init__(self, run_dir: str, rank: int = 0):
        self.run_dir = run_dir
        self.rank = rank
        self.enabled = (rank == 0 and bool(run_dir))

        if not self.enabled:
            return

        os.makedirs(self.run_dir, exist_ok=True)
        self.sync_events_path = os.path.join(self.run_dir, "sync_events.csv")
        self.decisions_path = os.path.join(self.run_dir, "controller_decisions.csv")
        self.epoch_metrics_path = os.path.join(self.run_dir, "epoch_metrics.csv")
        self.runtime_breakdown_path = os.path.join(self.run_dir, "runtime_breakdown.csv")
        self.metadata_path = os.path.join(self.run_dir, "run_metadata.json")

        self._init_files()

    def _init_files(self) -> None:
        if not os.path.exists(self.sync_events_path):
            with open(self.sync_events_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round_idx",
                    "epoch",
                    "global_step",
                    "sync_reason",
                    "is_forced",
                    "actual_local_steps",
                    "planned_h",
                    "samples_since_sync",
                    "total_samples_since_sync",
                    "v_t",
                    "q_t",
                    "m_t",
                    "norm_w_bar_sq",
                    "legacy_drift",
                    "compute_time_sec",
                    "sync_time_sec",
                    "overhead_sec",
                    "communicated_bytes",
                ])

        if not os.path.exists(self.decisions_path):
            with open(self.decisions_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round_idx",
                    "epoch",
                    "global_step",
                    "current_h",
                    "next_h",
                    "decision",
                    "reason",
                    "controller_type",
                    "budget",
                    "feasible_candidates",
                    "candidate_predictions",
                    "candidate_costs",
                    "lagrangian_scores",
                    "dual_variable",
                ])

        if not os.path.exists(self.epoch_metrics_path):
            with open(self.epoch_metrics_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "val_loss",
                    "val_acc",
                    "epoch_time_sec",
                    "train_time_sec",
                    "val_time_sec",
                    "learning_rate",
                    "cumulative_comm_rounds",
                    "effective_throughput_img_per_sec",
                ])

        if not os.path.exists(self.runtime_breakdown_path):
            with open(self.runtime_breakdown_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "total_elapsed_sec",
                    "total_compute_sec",
                    "total_sync_sec",
                    "total_val_sec",
                    "total_overhead_sec",
                    "sync_pct",
                    "compute_pct",
                ])

    def log_sync_event(self, record: SyncRecord) -> None:
        if not self.enabled:
            return
        m = record.metrics
        with open(self.sync_events_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.round_idx,
                record.epoch,
                record.global_step,
                record.sync_reason,
                int(record.is_forced),
                record.actual_local_steps,
                record.planned_h,
                record.samples_since_sync,
                record.total_samples_since_sync,
                f"{m.v_t:.8e}",
                f"{m.q_t:.8e}",
                f"{m.m_t:.8e}",
                f"{m.norm_w_bar_sq:.8e}",
                f"{m.legacy_drift:.8e}",
                f"{record.compute_time_sec:.4f}",
                f"{record.sync_time_sec:.4f}",
                f"{record.overhead_sec:.6f}",
                record.communicated_bytes,
            ])

    def log_decision(
        self,
        record: SyncRecord,
        decision: AdaptiveDecision,
        controller_type: str,
        budget: float,
        feasible_candidates: Optional[list] = None,
        candidate_predictions: Optional[dict] = None,
        candidate_costs: Optional[dict] = None,
        lagrangian_scores: Optional[dict] = None,
        dual_variable: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        with open(self.decisions_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.round_idx,
                record.epoch,
                record.global_step,
                record.planned_h,
                decision.next_h,
                decision.decision,
                decision.reason,
                controller_type,
                f"{budget:.8e}" if budget is not None else "",
                json.dumps(feasible_candidates if feasible_candidates is not None else []),
                json.dumps(candidate_predictions if candidate_predictions is not None else {}),
                json.dumps(candidate_costs if candidate_costs is not None else {}),
                json.dumps(lagrangian_scores if lagrangian_scores is not None else {}),
                f"{dual_variable:.6e}" if dual_variable is not None else "",
            ])

    def log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        epoch_time_sec: float,
        train_time_sec: float,
        val_time_sec: float,
        learning_rate: float,
        cumulative_comm_rounds: int,
        effective_throughput: float,
    ) -> None:
        if not self.enabled:
            return
        with open(self.epoch_metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.4f}",
                f"{train_acc:.4f}",
                f"{val_loss:.4f}",
                f"{val_acc:.4f}",
                f"{epoch_time_sec:.2f}",
                f"{train_time_sec:.2f}",
                f"{val_time_sec:.2f}",
                f"{learning_rate:.6e}",
                cumulative_comm_rounds,
                f"{effective_throughput:.2f}",
            ])

    def log_runtime_breakdown(
        self,
        epoch: int,
        total_elapsed_sec: float,
        total_compute_sec: float,
        total_sync_sec: float,
        total_val_sec: float,
        total_overhead_sec: float,
    ) -> None:
        if not self.enabled:
            return
        denom = max(1e-6, total_elapsed_sec)
        sync_pct = (total_sync_sec / denom) * 100.0
        compute_pct = (total_compute_sec / denom) * 100.0
        with open(self.runtime_breakdown_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{total_elapsed_sec:.2f}",
                f"{total_compute_sec:.2f}",
                f"{total_sync_sec:.2f}",
                f"{total_val_sec:.2f}",
                f"{total_overhead_sec:.4f}",
                f"{sync_pct:.2f}",
                f"{compute_pct:.2f}",
            ])

    def save_run_metadata(self, metadata: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

