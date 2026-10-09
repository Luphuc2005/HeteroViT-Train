"""V1 — Consensus-Budget Adaptive-H (CBA-H) Controller.

Mathematical Formulation:
  For each candidate H in {40, 80, 120, 150}:
    1. Consensus Error Extrapolation:
         V_hat_t(H) = V_t * (H / h_t)^p     (for raw_v metric)
       or
         Q_hat_t(H) = Q_t * (H / h_t)^p     (for normalized_q metric)
       Default growth exponent: p = 2.0 (conservative empirical extrapolation).

    2. Runtime Cost Prediction:
         C_t(H) = c_t + s_t / H  (or epoch-aware cost via CostEstimator).

    3. Discrete Constrained Optimization:
         minimize C_t(H)
         subject to:
           metric_hat_t(H) <= Budget B
           H in {40, 80, 120, 150}

    4. Adjacent Level Restriction (default enabled):
         |index(H_next) - index(H_current)| <= 1

    5. Safety Fallback:
         If no candidate is feasible (or on numerical invalidity),
         choose H_min = 40 and record budget violation.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from .controller_base import AdaptiveHControllerBase, SyncRecord, AdaptiveDecision
from .cost_estimator import CostEstimator


class ConsensusBudgetController(AdaptiveHControllerBase):
    """V1 Consensus-Budget Adaptive-H (CBA-H) Controller."""

    def __init__(
        self,
        candidates: List[int] = [40, 80, 120, 150],
        initial_h: int = 40,
        budget: float = 0.05,
        metric_type: str = "normalized_q",  # 'normalized_q' or 'raw_v'
        growth_exponent: float = 2.0,
        steps_per_epoch: int = 150,
        adjacent_step_only: bool = True,
        cost_estimator: Optional[CostEstimator] = None,
        epoch_aware_cost: bool = True,
    ):
        super().__init__(
            candidates=candidates,
            initial_h=initial_h,
            steps_per_epoch=steps_per_epoch,
            adjacent_step_only=adjacent_step_only,
        )
        if budget <= 0.0 or not math.isfinite(budget):
            raise ValueError(f"Budget must be finite and > 0, got {budget}")
        if metric_type not in ("normalized_q", "raw_v"):
            raise ValueError(f"Unknown metric_type: '{metric_type}'. Expected 'normalized_q' or 'raw_v'.")
        if growth_exponent <= 0.0 or not math.isfinite(growth_exponent):
            raise ValueError(f"Growth exponent must be finite and > 0, got {growth_exponent}")

        self.budget = float(budget)
        self.metric_type = metric_type
        self.growth_exponent = float(growth_exponent)
        self.cost_estimator = cost_estimator or CostEstimator(
            steps_per_epoch=steps_per_epoch,
            epoch_aware_cost=epoch_aware_cost,
        )

        self.last_observed_metric: Optional[float] = None
        self.last_observed_steps: Optional[int] = None
        self.budget_violation_count = 0
        self.valid_observations_count = 0

    def predict_metric(self, h: int, observed_metric: float, observed_steps: int) -> float:
        """Extrapolates consensus metric from observed block using power-law growth model.
        
        Note: This is an empirical extrapolation model, NOT a mathematical upper bound theorem.
        """
        if observed_steps <= 0:
            raise ValueError(f"observed_steps must be > 0, got {observed_steps}")
        ratio = float(h) / float(observed_steps)
        return float(observed_metric) * (ratio ** self.growth_exponent)

    def observe(self, record: SyncRecord) -> None:
        """Updates internal cost estimator and records consensus metric observation."""
        m = record.metrics
        metric_val = m.q_t if self.metric_type == "normalized_q" else m.v_t

        if not math.isfinite(metric_val) or metric_val < 0.0:
            # Handle invalid observation safely
            return

        # Update cost estimator
        self.cost_estimator.update(
            actual_local_steps=record.actual_local_steps,
            compute_time_sec=record.compute_time_sec,
            sync_time_sec=record.sync_time_sec,
            overhead_sec=record.overhead_sec,
        )

        # Only update metric extrapolation anchor on full planned blocks
        if not record.is_forced or record.actual_local_steps >= record.planned_h:
            self.last_observed_metric = metric_val
            self.last_observed_steps = record.actual_local_steps
            self.valid_observations_count += 1

    def propose_next_h(self, steps_remaining_in_epoch: Optional[int] = None) -> AdaptiveDecision:
        """Solves constrained optimization problem over candidates."""
        # Cold start safety: keep initial_h until first valid full block is observed
        if self.last_observed_metric is None or self.last_observed_steps is None or self.valid_observations_count == 0:
            return AdaptiveDecision(
                next_h=self.current_h,
                decision="KEEP",
                reason="Cold start: awaiting first full block observation.",
                metadata={"cold_start": True},
            )

        obs_metric = self.last_observed_metric
        obs_steps = self.last_observed_steps

        # 1. Predict consensus metric for each candidate
        candidate_predictions = {}
        for h in self.candidates:
            pred = self.predict_metric(h, obs_metric, obs_steps)
            candidate_predictions[h] = pred

        # 2. Predict cost per step for each candidate
        candidate_costs = self.cost_estimator.predict_all(
            self.candidates,
            steps_remaining=steps_remaining_in_epoch,
        )

        # 3. Determine feasibility
        feasible_candidates = [
            h for h in self.candidates
            if candidate_predictions[h] <= self.budget
        ]

        # 4. Selection
        if not feasible_candidates:
            # Budget violation: Fallback to minimum H
            self.budget_violation_count += 1
            target_h = self.candidates[0]
            reason = (
                f"Budget violation: no candidate feasible under budget {self.budget:.4e}. "
                f"Minimum predicted metric: {min(candidate_predictions.values()):.4e}. "
                f"Falling back to H_min={target_h}."
            )
            raw_decision = "FALLBACK_MIN"
        else:
            # Select feasible candidate with minimum cost per step (largest feasible H)
            target_h = min(feasible_candidates, key=lambda h: (candidate_costs[h], -h))
            reason = (
                f"Selected H={target_h} from feasible candidates {feasible_candidates} "
                f"with predicted metric={candidate_predictions[target_h]:.4e} <= budget {self.budget:.4e}."
            )
            raw_decision = "SELECT_FEASIBLE"

        # 5. Apply adjacent transition limit
        next_h = self.clamp_transition(target_h)
        if next_h > self.current_h:
            decision_label = "INCREASE"
        elif next_h < self.current_h:
            decision_label = "DECREASE"
        else:
            decision_label = "KEEP"

        metadata = {
            "controller_type": "CBA-H (V1)",
            "metric_type": self.metric_type,
            "budget": self.budget,
            "growth_exponent": self.growth_exponent,
            "candidate_predictions": candidate_predictions,
            "candidate_costs": candidate_costs,
            "feasible_candidates": feasible_candidates,
            "target_h_before_clamp": target_h,
            "raw_decision": raw_decision,
        }

        return AdaptiveDecision(
            next_h=next_h,
            decision=decision_label,
            reason=reason,
            metadata=metadata,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_h": self.current_h,
            "sync_rounds": self.sync_rounds,
            "budget": self.budget,
            "metric_type": self.metric_type,
            "growth_exponent": self.growth_exponent,
            "last_observed_metric": self.last_observed_metric,
            "last_observed_steps": self.last_observed_steps,
            "budget_violation_count": self.budget_violation_count,
            "valid_observations_count": self.valid_observations_count,
            "cost_estimator": self.cost_estimator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_h = int(state.get("current_h", self.current_h))
        self.sync_rounds = int(state.get("sync_rounds", self.sync_rounds))
        self.budget = float(state.get("budget", self.budget))
        self.metric_type = str(state.get("metric_type", self.metric_type))
        self.growth_exponent = float(state.get("growth_exponent", self.growth_exponent))
        self.last_observed_metric = state.get("last_observed_metric", self.last_observed_metric)
        self.last_observed_steps = state.get("last_observed_steps", self.last_observed_steps)
        self.budget_violation_count = int(state.get("budget_violation_count", self.budget_violation_count))
        self.valid_observations_count = int(state.get("valid_observations_count", self.valid_observations_count))
        if "cost_estimator" in state:
            self.cost_estimator.load_state_dict(state["cost_estimator"])

