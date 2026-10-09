"""V3 — Primal-Dual Cost-Aware Adaptive-H (PDCA-H) Controller.

Mathematical Formulation:
  Objective:
    Minimize cumulative wall-clock time per training step subject to an online
    risk-budget surrogate constraint:
      minimize sum_t T_t(H_t)
      subject to: (1 / sum_t h_t) * sum_t h_t * R_t <= R_budget

  Lagrangian Relaxation:
    Dual multiplier mu_t >= 0 associated with the risk constraint.
    Lagrangian score for candidate H:
      J_t(H) = T_hat_t(H) + mu_t * R_hat_t(H)

    Selection policy:
      H_next = argmin_{H in H_candidates} J_t(H)

    Dual Subgradient Update:
      mu_{t+1} = clip(mu_t + eta_mu * violation_t, 0.0, mu_max)
      where violation_t = (h_t / h_ref) * (R_observed - R_budget)

  Risk Surrogate:
    R_hat_t(H) is the predicted normalized consensus error Q_hat_t(H),
    estimated using the online dynamics predictor from V2 (with fallback to V1
    prior to model identifiability).

  System Cost:
    T_hat_t(H) is the predicted wall-clock cost per useful step,
    estimated via CostEstimator (accounting for compute time, sync time, and
    epoch-boundary forced synchronizations).
"""

import math
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .controller_base import AdaptiveHControllerBase, SyncRecord, AdaptiveDecision
from .cost_estimator import CostEstimator
from .v2_online_dynamics import OnlineDriftDynamicsController


class PrimalDualCostAwareController(AdaptiveHControllerBase):
    """V3 Primal-Dual Cost-Aware Adaptive-H (PDCA-H) Controller."""

    def __init__(
        self,
        candidates: List[int] = [40, 80, 120, 150],
        initial_h: int = 40,
        risk_budget: float = 0.03,            # R_budget
        dual_step_size: float = 1.0,           # eta_mu
        dual_initial: float = 0.0,             # mu_0
        dual_max: Optional[float] = 50.0,      # mu_max upper cap
        metric_type: str = "normalized_q",     # 'normalized_q' or 'raw_v'
        weighting_mode: str = "step_weighted", # 'step_weighted' or 'block_weighted'
        steps_per_epoch: int = 150,
        adjacent_step_only: bool = True,
        cost_estimator: Optional[CostEstimator] = None,
        epoch_aware_cost: bool = True,
        # V2 dynamics sub-predictor parameters
        forgetting_factor: float = 0.98,
        uncertainty_multiplier: float = 1.0,
        min_distinct_h: int = 2,
        min_observations: int = 3,
        fallback_exponent: float = 2.0,
    ):
        super().__init__(
            candidates=candidates,
            initial_h=initial_h,
            steps_per_epoch=steps_per_epoch,
            adjacent_step_only=adjacent_step_only,
        )
        if risk_budget <= 0.0 or not math.isfinite(risk_budget):
            raise ValueError(f"risk_budget must be finite and > 0, got {risk_budget}")
        if dual_step_size <= 0.0 or not math.isfinite(dual_step_size):
            raise ValueError(f"dual_step_size must be finite and > 0, got {dual_step_size}")
        if dual_initial < 0.0 or not math.isfinite(dual_initial):
            raise ValueError(f"dual_initial must be non-negative, got {dual_initial}")
        if weighting_mode not in ("step_weighted", "block_weighted"):
            raise ValueError(f"Unknown weighting_mode: {weighting_mode}")

        self.risk_budget = float(risk_budget)
        self.dual_step_size = float(dual_step_size)
        self.dual_var = float(dual_initial)    # mu_t
        self.dual_max = float(dual_max) if dual_max is not None else None
        self.metric_type = metric_type
        self.weighting_mode = weighting_mode

        self.cost_estimator = cost_estimator or CostEstimator(
            steps_per_epoch=steps_per_epoch,
            epoch_aware_cost=epoch_aware_cost,
        )

        # Internal online drift dynamics predictor (from V2)
        self.dynamics_predictor = OnlineDriftDynamicsController(
            candidates=candidates,
            initial_h=initial_h,
            budget=risk_budget,
            metric_type=metric_type,
            fallback_exponent=fallback_exponent,
            forgetting_factor=forgetting_factor,
            uncertainty_multiplier=uncertainty_multiplier,
            min_distinct_h=min_distinct_h,
            min_observations=min_observations,
            steps_per_epoch=steps_per_epoch,
            adjacent_step_only=adjacent_step_only,
            cost_estimator=self.cost_estimator,
            epoch_aware_cost=epoch_aware_cost,
        )

        # Cumulative tracking
        self.cumulative_steps = 0
        self.cumulative_weighted_risk = 0.0
        self.last_observed_risk: Optional[float] = None
        self.last_constraint_violation: float = 0.0
        self.dual_clipping_events = 0
        self.budget_violations = 0

    def observe(self, record: SyncRecord) -> None:
        """Observes sync record, updates internal predictor, cost model, and dual variable."""
        # 1. Forward observation to dynamics predictor and cost estimator
        self.dynamics_predictor.observe(record)

        m = record.metrics
        r_observed = m.q_t if self.metric_type == "normalized_q" else m.v_t

        if not math.isfinite(r_observed) or r_observed < 0.0:
            return

        self.last_observed_risk = r_observed
        h_t = record.actual_local_steps

        # 2. Cumulative tracking
        self.cumulative_steps += h_t
        self.cumulative_weighted_risk += h_t * r_observed

        # 3. Dual update (only on valid full blocks to prevent epoch-end partial-block bias)
        if not record.is_forced or record.actual_local_steps >= record.planned_h:
            raw_violation = r_observed - self.risk_budget
            if raw_violation > 0:
                self.budget_violations += 1

            if self.weighting_mode == "step_weighted":
                # Scale violation by step weight relative to nominal candidate
                step_scale = float(h_t) / float(self.candidates[0])
                violation = step_scale * raw_violation
            else:
                violation = raw_violation

            self.last_constraint_violation = violation

            # Dual subgradient update: mu_{t+1} = max(0, mu_t + eta * violation)
            new_mu = self.dual_var + self.dual_step_size * violation
            new_mu = max(0.0, new_mu)

            if self.dual_max is not None and new_mu > self.dual_max:
                new_mu = self.dual_max
                self.dual_clipping_events += 1

            self.dual_var = new_mu

    def propose_next_h(self, steps_remaining_in_epoch: Optional[int] = None) -> AdaptiveDecision:
        """Computes next H by minimizing the Lagrangian score J(H) = T(H) + mu * R(H)."""
        # 1. Cold start / fallback check:
        # If dynamics predictor is not yet identified, fall back to safe conservative logic
        if not self.dynamics_predictor.is_identified:
            v2_decision = self.dynamics_predictor.propose_next_h(steps_remaining_in_epoch)
            metadata = dict(v2_decision.metadata)
            metadata.update({
                "controller_type": "PDCA-H (V3)",
                "identified": False,
                "dual_variable": self.dual_var,
                "risk_budget": self.risk_budget,
                "fallback_to_v2": True,
            })
            return AdaptiveDecision(
                next_h=v2_decision.next_h,
                decision=v2_decision.decision,
                reason=f"Predictor unready: Falling back to conservative dynamics. ({v2_decision.reason})",
                metadata=metadata,
            )

        # 2. Predict cost per step T_hat(H) for each candidate
        candidate_costs = self.cost_estimator.predict_all(
            self.candidates,
            steps_remaining=steps_remaining_in_epoch,
        )

        # 3. Predict risk surrogate R_hat(H) for each candidate (using upper prediction for safety)
        candidate_risks = {}
        candidate_uncertainties = {}
        for h in self.candidates:
            pred, unc, upper = self.dynamics_predictor.predict_candidate(h)
            candidate_risks[h] = upper
            candidate_uncertainties[h] = unc

        # 4. Compute Lagrangian scores J(H) = T_hat(H) + mu * R_hat(H)
        lagrangian_scores = {}
        for h in self.candidates:
            cost_h = candidate_costs[h]
            risk_h = candidate_risks[h]
            lagrangian_scores[h] = cost_h + self.dual_var * risk_h

        # 5. Optimize: H* = argmin_H J(H)
        target_h = min(self.candidates, key=lambda h: lagrangian_scores[h])

        # 6. Apply adjacent transition restriction
        next_h = self.clamp_transition(target_h)
        if next_h > self.current_h:
            decision_label = "INCREASE"
        elif next_h < self.current_h:
            decision_label = "DECREASE"
        else:
            decision_label = "KEEP"

        reason = (
            f"Lagrangian argmin: Selected H={target_h} (J={lagrangian_scores[target_h]:.4f}, "
            f"T={candidate_costs[target_h]:.4f}s/step, R={candidate_risks[target_h]:.4e}, "
            f"mu={self.dual_var:.4f}). Clamped next_h={next_h}."
        )

        metadata = {
            "controller_type": "PDCA-H (V3)",
            "identified": True,
            "dual_variable": self.dual_var,
            "risk_budget": self.risk_budget,
            "dual_step_size": self.dual_step_size,
            "candidate_costs": candidate_costs,
            "candidate_risks": candidate_risks,
            "candidate_uncertainties": candidate_uncertainties,
            "lagrangian_scores": lagrangian_scores,
            "target_h_before_clamp": target_h,
            "last_constraint_violation": self.last_constraint_violation,
            "cumulative_avg_risk": (
                self.cumulative_weighted_risk / max(1, self.cumulative_steps)
            ),
            "dual_clipping_events": self.dual_clipping_events,
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
            "dual_var": self.dual_var,
            "risk_budget": self.risk_budget,
            "dual_step_size": self.dual_step_size,
            "dual_max": self.dual_max,
            "metric_type": self.metric_type,
            "weighting_mode": self.weighting_mode,
            "cumulative_steps": self.cumulative_steps,
            "cumulative_weighted_risk": self.cumulative_weighted_risk,
            "last_observed_risk": self.last_observed_risk,
            "last_constraint_violation": self.last_constraint_violation,
            "dual_clipping_events": self.dual_clipping_events,
            "budget_violations": self.budget_violations,
            "dynamics_predictor": self.dynamics_predictor.state_dict(),
            "cost_estimator": self.cost_estimator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_h = int(state.get("current_h", self.current_h))
        self.sync_rounds = int(state.get("sync_rounds", self.sync_rounds))
        self.dual_var = float(state.get("dual_var", self.dual_var))
        self.risk_budget = float(state.get("risk_budget", self.risk_budget))
        self.dual_step_size = float(state.get("dual_step_size", self.dual_step_size))
        self.dual_max = float(state["dual_max"]) if state.get("dual_max") is not None else None
        self.metric_type = str(state.get("metric_type", self.metric_type))
        self.weighting_mode = str(state.get("weighting_mode", self.weighting_mode))
        self.cumulative_steps = int(state.get("cumulative_steps", self.cumulative_steps))
        self.cumulative_weighted_risk = float(state.get("cumulative_weighted_risk", self.cumulative_weighted_risk))
        self.last_observed_risk = state.get("last_observed_risk", self.last_observed_risk)
        self.last_constraint_violation = float(state.get("last_constraint_violation", self.last_constraint_violation))
        self.dual_clipping_events = int(state.get("dual_clipping_events", self.dual_clipping_events))
        self.budget_violations = int(state.get("budget_violations", self.budget_violations))
        if "dynamics_predictor" in state:
            self.dynamics_predictor.load_state_dict(state["dynamics_predictor"])
        if "cost_estimator" in state:
            self.cost_estimator.load_state_dict(state["cost_estimator"])

