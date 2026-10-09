"""
Drift-Adaptive-H Controller for Local SGD / Periodic Model Parameter Averaging.

Implements V1:
  - Online adjustment of synchronization period H_t based on normalized model drift D_t.
  - Candidate set H in {40, 80, 120, 150} (or user-configured ascending positive integers).
  - Two-threshold discrete controller: tau_low < tau_high.
  - At most one discrete step adjustment per controller update.
  - Pure functions for validation and decision making.
  - Minimal scalar MPI communication overhead (1-2 floats, no secondary full-model Allreduce).
"""

import os
import csv
import time
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


def validate_adaptive_h_config(cfg: Dict[str, Any]) -> None:
    """Validates Drift-Adaptive-H configuration strictly.
    
    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    candidates = cfg.get("candidates", [40, 80, 120, 150])
    if not candidates or not isinstance(candidates, (list, tuple)):
        raise ValueError("adaptive_h.candidates must be a non-empty list/tuple of positive integers.")

    candidates_list = [int(x) for x in candidates]
    if any(c <= 0 for c in candidates_list):
        raise ValueError(f"All candidate H values must be > 0, got: {candidates_list}")

    if len(candidates_list) != len(set(candidates_list)):
        raise ValueError(f"Candidate H values must be unique, got: {candidates_list}")

    if candidates_list != sorted(candidates_list):
        raise ValueError(f"Candidate H values must be sorted ascending, got: {candidates_list}")

    initial_h = int(cfg.get("initial_h", candidates_list[0]))
    if initial_h not in candidates_list:
        raise ValueError(f"initial_h ({initial_h}) must belong to candidates ({candidates_list})")

    tau_low = float(cfg.get("tau_low", 0.02))
    tau_high = float(cfg.get("tau_high", 0.05))
    epsilon = float(cfg.get("epsilon", 1.0e-12))

    if not np.isfinite(tau_low) or tau_low < 0.0:
        raise ValueError(f"tau_low must be finite and >= 0, got: {tau_low}")

    if not np.isfinite(tau_high):
        raise ValueError(f"tau_high must be finite, got: {tau_high}")

    if tau_low >= tau_high:
        raise ValueError(f"tau_low ({tau_low}) must be strictly less than tau_high ({tau_high})")

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and > 0, got: {epsilon}")


def choose_next_h(
    current_h: int,
    drift: float,
    candidates: List[int],
    tau_low: float,
    tau_high: float,
) -> Tuple[int, str]:
    """Pure controller function mapping (current_h, drift) -> (next_h, decision).
    
    Rules:
      - drift < tau_low: next_larger(current_h)
      - tau_low <= drift <= tau_high: current_h (KEEP)
      - drift > tau_high: next_smaller(current_h)
      
    Boundaries:
      - next_larger(max(candidates)) = max(candidates)
      - next_smaller(min(candidates)) = min(candidates)
      - At most one discrete step transition per invocation.
      
    Args:
        current_h: Current synchronization period.
        drift: Aggregated global normalized drift D_t.
        candidates: Sorted list of candidate periods.
        tau_low: Lower drift threshold.
        tau_high: Upper drift threshold.
        
    Returns:
        (next_h, decision) where decision in {"INCREASE", "KEEP", "DECREASE"}.
    """
    if np.isnan(drift) or np.isinf(drift):
        raise ValueError(f"Invalid drift value: {drift}. Drift must be a finite, non-NaN number.")

    if current_h not in candidates:
        raise ValueError(f"current_h ({current_h}) is not in candidates ({candidates})")

    curr_idx = candidates.index(current_h)

    if drift < tau_low:
        next_idx = min(curr_idx + 1, len(candidates) - 1)
        next_h = candidates[next_idx]
        decision = "INCREASE" if next_h > current_h else "KEEP"
    elif drift > tau_high:
        next_idx = max(curr_idx - 1, 0)
        next_h = candidates[next_idx]
        decision = "DECREASE" if next_h < current_h else "KEEP"
    else:
        next_h = current_h
        decision = "KEEP"

    return next_h, decision


def compute_normalized_local_drift(
    flat_w: np.ndarray,
    global_flat_w: np.ndarray,
    epsilon: float = 1.0e-12,
) -> float:
    """Computes normalized local model drift:
    
        d_i = ||W_i - W_bar||_2 / (||W_bar||_2 + epsilon)
        
    Computed locally on each rank reusing existing parameter buffers.
    """
    diff_l2 = float(np.linalg.norm(flat_w - global_flat_w))
    global_l2 = float(np.linalg.norm(global_flat_w))
    return diff_l2 / (global_l2 + epsilon)


def aggregate_global_drift(
    comm,
    local_drift: float,
    samples_since_sync: int,
    policy: str,
    world_size: int,
    epsilon: float = 1.0e-12,
) -> float:
    """Aggregates global drift D_t across ranks via a single scalar Allreduce.
    
    If sample_weighted:
        D_t = sum_i(n_i * d_i) / sum_i(n_i)
    If uniform:
        D_t = (1 / N) * sum_i(d_i)
        
    Communication overhead:
      - sample_weighted: 2 float64 numbers (16 bytes).
      - uniform: 1 float64 number (8 bytes).
    """
    from mpi4py import MPI

    if policy == "sample_weighted":
        n_i = float(max(0, samples_since_sync))
        v_i = n_i * float(local_drift)
        drift_buf = np.array([v_i, n_i], dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, drift_buf, op=MPI.SUM)
        total_weighted_drift = drift_buf[0]
        total_samples = drift_buf[1]
        if total_samples > 0.0:
            return float(total_weighted_drift / total_samples)
        else:
            # Fallback if total samples is 0
            buf = np.array([float(local_drift)], dtype=np.float64)
            comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
            return float(buf[0] / float(world_size))
    elif policy == "uniform":
        buf = np.array([float(local_drift)], dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, buf, op=MPI.SUM)
        return float(buf[0] / float(world_size))
    else:
        raise ValueError(f"Unknown averaging policy: '{policy}'. Supported: 'sample_weighted', 'uniform'")


class DriftAdaptiveHController:
    """Stateful coordinator for Drift-Adaptive-H Local SGD.
    
    Manages:
      - current_h, initial_h, candidates, thresholds, epsilon.
      - sync_round counter.
      - Machine-readable trace logging to adaptive_h_trace.csv.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Optional[str] = None,
        rank: int = 0,
        comm = None,
    ):
        validate_adaptive_h_config(config)

        self.candidates = [int(x) for x in config.get("candidates", [40, 80, 120, 150])]
        self.initial_h = int(config.get("initial_h", self.candidates[0]))
        self.tau_low = float(config.get("tau_low", 0.02))
        self.tau_high = float(config.get("tau_high", 0.05))
        self.epsilon = float(config.get("epsilon", 1.0e-12))

        self.current_h = self.initial_h
        self.sync_round = 0
        self.last_drift = 0.0
        self.rank = rank
        self.comm = comm

        self.trace_csv_path = None
        if self.rank == 0 and run_dir:
            self.trace_csv_path = os.path.join(run_dir, "adaptive_h_trace.csv")
            if not os.path.exists(self.trace_csv_path):
                with open(self.trace_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "sync_round",
                        "epoch",
                        "global_step",
                        "current_h",
                        "next_h",
                        "drift",
                        "tau_low",
                        "tau_high",
                        "decision",
                        "sync_reason",
                        "actual_local_steps",
                        "planned_h",
                        "controller_updated",
                        "samples_since_sync",
                        "compute_time_since_last_sync",
                        "sync_time_sec",
                        "drift_compute_time_sec",
                    ])

    def process_sync_event(
        self,
        epoch: int,
        global_step: int,
        sync_reason: str,
        actual_local_steps: int,
        samples_since_sync: int,
        global_drift: float,
        compute_time_since_last_sync: float,
        sync_time_sec: float,
        drift_compute_time_sec: float,
    ) -> Tuple[int, str, bool]:
        """Processes a synchronization event and determines next H.
        
        Args:
            epoch: Current epoch.
            global_step: Total local optimizer steps completed by this rank.
            sync_reason: 'H_REACHED', 'EPOCH_END', or 'TRAIN_END'.
            actual_local_steps: Local steps completed since previous sync.
            samples_since_sync: Samples processed by this rank since previous sync.
            global_drift: Normalized global model drift D_t.
            compute_time_since_last_sync: Wall-clock seconds spent computing local steps.
            sync_time_sec: Wall-clock seconds spent in model averaging.
            drift_compute_time_sec: Wall-clock seconds spent measuring drift.
            
        Returns:
            (next_h, decision, controller_updated)
        """
        planned_h = self.current_h
        self.sync_round += 1

        # Only update controller on full planned blocks
        if sync_reason == "H_REACHED" and actual_local_steps == planned_h:
            t_ctl_start = time.perf_counter()
            next_h, decision = choose_next_h(
                current_h=self.current_h,
                drift=global_drift,
                candidates=self.candidates,
                tau_low=self.tau_low,
                tau_high=self.tau_high,
            )
            controller_compute_time_sec = time.perf_counter() - t_ctl_start
            controller_updated = True
        else:
            next_h = self.current_h
            decision = "SKIP_PARTIAL"
            controller_updated = False
            controller_compute_time_sec = 0.0

        # Log trace row on Rank 0
        if self.rank == 0 and self.trace_csv_path:
            with open(self.trace_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.sync_round,
                    epoch,
                    global_step,
                    self.current_h,
                    next_h,
                    f"{global_drift:.6e}",
                    f"{self.tau_low:.6f}",
                    f"{self.tau_high:.6f}",
                    decision,
                    sync_reason,
                    actual_local_steps,
                    planned_h,
                    int(controller_updated),
                    samples_since_sync,
                    f"{compute_time_since_last_sync:.4f}",
                    f"{sync_time_sec:.4f}",
                    f"{drift_compute_time_sec:.6f}",
                ])

        old_h = self.current_h
        self.current_h = next_h
        self.last_drift = global_drift

        return next_h, decision, controller_updated

