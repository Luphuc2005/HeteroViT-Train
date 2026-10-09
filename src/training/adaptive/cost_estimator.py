"""Cost Estimator and Epoch-Aware Runtime Models for Adaptive Local SGD.

Models:
  1. Simplified Analytical Cost:
       C_t(H) = c_t + s_t / H
     where c_t is compute time per step, and s_t is synchronization time.

  2. Epoch-Aware Runtime Cost:
       Given steps_per_epoch S (e.g. 150) or steps remaining R:
       The number of sync events N_sync(R, H) accounts for regular full blocks
       and the forced sync at epoch boundary:
         N_sync(R, H) = ceil(R / H)
       Total cost = R * c_t + N_sync(R, H) * (s_t + o_t)
       Cost per step = c_t + (N_sync(R, H) / R) * (s_t + o_t)

  3. Exponential Moving Average (EMA) smoothing for observed step compute time,
     sync time, and controller overhead.
"""

import math
from typing import Dict, Any, Optional, List


def calculate_epoch_sync_count(steps_per_epoch: int, h: int) -> int:
    """Calculates the exact number of model synchronizations in an epoch of S steps.
    
    If S % H == 0:
        Exactly S // H sync rounds.
    Else:
        (S // H) full-block syncs + 1 forced partial-block sync at epoch end.
        Total = ceil(S / H).
    """
    if h <= 0:
        raise ValueError(f"H must be > 0, got {h}")
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
    return math.ceil(steps_per_epoch / h)


class EpochAwareCostModel:
    """Static calculations for epoch-aware communication counts and step costs."""

    @staticmethod
    def sync_count(steps: int, h: int) -> int:
        return calculate_epoch_sync_count(steps, h)

    @staticmethod
    def epoch_total_time(
        steps_per_epoch: int,
        h: int,
        compute_time_per_step: float,
        sync_time_per_round: float,
        controller_overhead_sec: float = 0.0,
    ) -> float:
        """Computes total expected wall-clock seconds for an epoch with interval H."""
        n_sync = calculate_epoch_sync_count(steps_per_epoch, h)
        total_compute = steps_per_epoch * compute_time_per_step
        total_sync = n_sync * (sync_time_per_round + controller_overhead_sec)
        return total_compute + total_sync

    @staticmethod
    def cost_per_step(
        h: int,
        compute_time_per_step: float,
        sync_time_per_round: float,
        steps_per_epoch: Optional[int] = None,
        controller_overhead_sec: float = 0.0,
        epoch_aware: bool = False,
    ) -> float:
        """Returns estimated wall-clock cost per useful training step."""
        overheads = sync_time_per_round + controller_overhead_sec
        if epoch_aware and steps_per_epoch is not None and steps_per_epoch > 0:
            n_sync = calculate_epoch_sync_count(steps_per_epoch, h)
            return compute_time_per_step + (n_sync / steps_per_epoch) * overheads
        else:
            return compute_time_per_step + (overheads / float(h))


class CostEstimator:
    """Stateful online estimator tracking compute time, sync time, and overheads via EMA."""

    def __init__(
        self,
        steps_per_epoch: int = 150,
        ema_alpha: float = 0.2,
        initial_compute_step_sec: float = 0.25,
        initial_sync_sec: float = 1.85,
        initial_overhead_sec: float = 0.005,
        epoch_aware_cost: bool = True,
    ):
        self.steps_per_epoch = int(steps_per_epoch)
        self.ema_alpha = float(ema_alpha)
        self.epoch_aware_cost = bool(epoch_aware_cost)

        self.compute_step_sec = float(initial_compute_step_sec)
        self.sync_sec = float(initial_sync_sec)
        self.overhead_sec = float(initial_overhead_sec)
        self.observations_count = 0

    def update(
        self,
        actual_local_steps: int,
        compute_time_sec: float,
        sync_time_sec: float,
        overhead_sec: float = 0.0,
    ) -> None:
        """Updates EMA estimates with fresh observation from a completed sync block."""
        if actual_local_steps > 0 and compute_time_sec > 0:
            observed_step_compute = compute_time_sec / float(actual_local_steps)
            if self.observations_count == 0:
                self.compute_step_sec = observed_step_compute
            else:
                self.compute_step_sec = (
                    self.ema_alpha * observed_step_compute
                    + (1.0 - self.ema_alpha) * self.compute_step_sec
                )

        if sync_time_sec > 0:
            if self.observations_count == 0:
                self.sync_sec = sync_time_sec
            else:
                self.sync_sec = (
                    self.ema_alpha * sync_time_sec
                    + (1.0 - self.ema_alpha) * self.sync_sec
                )

        if overhead_sec > 0:
            if self.observations_count == 0:
                self.overhead_sec = overhead_sec
            else:
                self.overhead_sec = (
                    self.ema_alpha * overhead_sec
                    + (1.0 - self.ema_alpha) * self.overhead_sec
                )

        self.observations_count += 1

    def predict_cost_per_step(
        self,
        h: int,
        steps_remaining: Optional[int] = None,
        epoch_aware: Optional[bool] = None,
    ) -> float:
        """Predicts wall-clock time per step for candidate interval H."""
        use_epoch_aware = self.epoch_aware_cost if epoch_aware is None else epoch_aware
        horizon = steps_remaining if steps_remaining is not None else self.steps_per_epoch
        return EpochAwareCostModel.cost_per_step(
            h=h,
            compute_time_per_step=self.compute_step_sec,
            sync_time_per_round=self.sync_sec,
            steps_per_epoch=horizon,
            controller_overhead_sec=self.overhead_sec,
            epoch_aware=use_epoch_aware,
        )

    def predict_all(
        self,
        candidates: List[int],
        steps_remaining: Optional[int] = None,
        epoch_aware: Optional[bool] = None,
    ) -> Dict[int, float]:
        """Returns predicted cost per step for each candidate H."""
        return {
            h: self.predict_cost_per_step(h, steps_remaining=steps_remaining, epoch_aware=epoch_aware)
            for h in candidates
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "compute_step_sec": self.compute_step_sec,
            "sync_sec": self.sync_sec,
            "overhead_sec": self.overhead_sec,
            "observations_count": self.observations_count,
            "steps_per_epoch": self.steps_per_epoch,
            "ema_alpha": self.ema_alpha,
            "epoch_aware_cost": self.epoch_aware_cost,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.compute_step_sec = float(state.get("compute_step_sec", self.compute_step_sec))
        self.sync_sec = float(state.get("sync_sec", self.sync_sec))
        self.overhead_sec = float(state.get("overhead_sec", self.overhead_sec))
        self.observations_count = int(state.get("observations_count", self.observations_count))
        self.steps_per_epoch = int(state.get("steps_per_epoch", self.steps_per_epoch))
        self.ema_alpha = float(state.get("ema_alpha", self.ema_alpha))
        self.epoch_aware_cost = bool(state.get("epoch_aware_cost", self.epoch_aware_cost))

