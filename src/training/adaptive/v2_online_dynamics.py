"""V2 — Online Drift-Dynamics Adaptive-H (ODD-H) Controller.

Mathematical Formulation:
  Consensus error dynamics model:
    V_hat_t(H) = a_t * H^(p_t)
  Log-linear formulation:
    log(metric_t + epsilon_v) = b_t + p_t * log(h_t) + e_t
  where:
    theta_t = [b_t, p_t]^T
    x_t = [1.0, log(h_t)]^T
    b_t = log(a_t)
    p_t = online learned drift-growth exponent
    e_t = prediction residual

Algorithm Components:
  1. 2-Parameter Recursive Least Squares (RLS):
       Gain: k_t = P_{t-1} x_t / (lambda + x_t^T P_{t-1} x_t)
       State: theta_t = theta_{t-1} + k_t * e_t
       Covariance: P_t = (P_{t-1} - k_t x_t^T P_{t-1}) / lambda
       Forgetting factor lambda in (0, 1] (default 0.98).

  2. Identifiability Requirement:
       If H does not vary, p_t cannot be identified.
       Track distinct full-block H levels observed.
       If distinct_h < min_distinct_h (default 2) or n_obs < min_observations (default 3),
       flag predictor as NOT READY and fall back to V1 fixed-exponent predictor.

  3. Exponent Bounds and Clipping:
       Clip p_t to [p_min, p_max] (default [0.1, 4.0]).
       Symmetrize P_t at each step.

  4. Prediction Uncertainty:
       sigma_log(H) = sqrt(max(0, x(H)^T P_t x(H) * sigma_residual^2))
       Upper prediction:
         log_upper = x(H)^T theta_t + kappa * sigma_log(H)
         U_t(H) = exp(log_upper) - epsilon_v

  5. Constrained Optimization:
       minimize Cost(H)
       subject to:
         U_t(H) <= Budget B
"""

import math
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

from .controller_base import AdaptiveHControllerBase, SyncRecord, AdaptiveDecision
from .cost_estimator import CostEstimator


class OnlineDriftDynamicsController(AdaptiveHControllerBase):
    """V2 Online Drift-Dynamics Adaptive-H (ODD-H) Controller."""

    def __init__(
        self,
        candidates: List[int] = [40, 80, 120, 150],
        initial_h: int = 40,
        budget: float = 0.05,
        metric_type: str = "normalized_q",  # 'normalized_q' or 'raw_v'
        fallback_exponent: float = 2.0,
        forgetting_factor: float = 0.98,     # lambda_RLS
        uncertainty_multiplier: float = 1.0, # kappa
        min_distinct_h: int = 2,
        min_observations: int = 3,
        p_bounds: Tuple[float, float] = (0.1, 4.0),
        epsilon_v: float = 1.0e-8,
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
        if forgetting_factor <= 0.0 or forgetting_factor > 1.0:
            raise ValueError(f"Forgetting factor must be in (0, 1], got {forgetting_factor}")
        if uncertainty_multiplier < 0.0:
            raise ValueError(f"uncertainty_multiplier must be >= 0, got {uncertainty_multiplier}")

        self.budget = float(budget)
        self.metric_type = metric_type
        self.fallback_exponent = float(fallback_exponent)
        self.lambda_rls = float(forgetting_factor)
        self.kappa = float(uncertainty_multiplier)
        self.min_distinct_h = int(min_distinct_h)
        self.min_observations = int(min_observations)
        self.p_min, self.p_max = float(p_bounds[0]), float(p_bounds[1])
        self.epsilon_v = float(epsilon_v)

        self.cost_estimator = cost_estimator or CostEstimator(
            steps_per_epoch=steps_per_epoch,
            epoch_aware_cost=epoch_aware_cost,
        )

        # RLS State: theta = [b, p]^T, where b = log(a), p = drift exponent
        # Initialize with prior: p = fallback_exponent, b calibrated so metric(40) ~ 0.01
        init_b = math.log(0.01 + self.epsilon_v) - self.fallback_exponent * math.log(float(initial_h))
        self.theta = np.array([init_b, self.fallback_exponent], dtype=np.float64)
        # Covariance matrix P: initial uncertainty
        self.P = np.eye(2, dtype=np.float64) * 10.0

        # Residual variance tracking (EMA)
        self.residual_var = 1.0e-4
        self.last_residual = 0.0

        # Observation tracking
        self.observed_h_levels: Set[int] = set()
        self.total_observations = 0
        self.clipping_events = 0
        self.fallback_count = 0
        self.budget_violation_count = 0

        self.last_observed_metric: Optional[float] = None
        self.last_observed_steps: Optional[int] = None

    @property
    def is_identified(self) -> bool:
        """Checks identifiability conditions: sufficient observations and variation in H."""
        return (
            len(self.observed_h_levels) >= self.min_distinct_h
            and self.total_observations >= self.min_observations
        )

    def _update_rls(self, h: int, metric_val: float) -> float:
        """Executes one step of 2-parameter RLS in log domain."""
        y = math.log(max(1.0e-12, metric_val) + self.epsilon_v)
        x = np.array([1.0, math.log(float(h))], dtype=np.float64)

        # Predicted log metric
        y_hat = float(np.dot(x, self.theta))
        residual = y - y_hat
        self.last_residual = residual

        # RLS Gain
        Px = np.dot(self.P, x)
        denom = self.lambda_rls + float(np.dot(x, Px))
        if denom <= 1.0e-12 or not np.isfinite(denom):
            # Singular step guard
            return residual

        k = Px / denom

        # Parameter update
        self.theta = self.theta + k * residual

        # Covariance update
        self.P = (self.P - np.outer(k, np.dot(x, self.P))) / self.lambda_rls
        # Symmetrize and regularize P
        self.P = 0.5 * (self.P + self.P.T)
        self.P += np.eye(2, dtype=np.float64) * 1.0e-6

        # Exponent clipping
        p_val = self.theta[1]
        if p_val < self.p_min or p_val > self.p_max:
            self.theta[1] = np.clip(p_val, self.p_min, self.p_max)
            self.clipping_events += 1

        # Residual variance EMA
        self.residual_var = 0.8 * self.residual_var + 0.2 * (residual ** 2)
        return residual

    def observe(self, record: SyncRecord) -> None:
        """Observes telemetry and updates RLS model if block is full and valid."""
        m = record.metrics
        metric_val = m.q_t if self.metric_type == "normalized_q" else m.v_t

        if not math.isfinite(metric_val) or metric_val < 0.0:
            return

        self.cost_estimator.update(
            actual_local_steps=record.actual_local_steps,
            compute_time_sec=record.compute_time_sec,
            sync_time_sec=record.sync_time_sec,
            overhead_sec=record.overhead_sec,
        )

        if not record.is_forced or record.actual_local_steps >= record.planned_h:
            h = record.actual_local_steps
            self.observed_h_levels.add(h)
            self.total_observations += 1
            self.last_observed_metric = metric_val
            self.last_observed_steps = h

            self._update_rls(h, metric_val)

    def predict_candidate(self, h: int) -> Tuple[float, float, float]:
        """Predicts (metric_hat, uncertainty, upper_prediction) for candidate H."""
        x = np.array([1.0, math.log(float(h))], dtype=np.float64)
        log_hat = float(np.dot(x, self.theta))
        pred_val = max(0.0, math.exp(np.clip(log_hat, -30.0, 10.0)) - self.epsilon_v)

        # Parameter covariance-based variance in log space
        var_log = float(np.dot(x, np.dot(self.P, x))) * max(1.0e-6, self.residual_var)
        sigma_log = math.sqrt(max(0.0, var_log))

        if self.kappa > 0.0:
            upper_log = np.clip(log_hat + self.kappa * sigma_log, -30.0, 10.0)
            upper_val = max(0.0, math.exp(upper_log) - self.epsilon_v)
            uncertainty = upper_val - pred_val
        else:
            uncertainty = 0.0
            upper_val = pred_val

        return pred_val, uncertainty, upper_val

    def propose_next_h(self, steps_remaining_in_epoch: Optional[int] = None) -> AdaptiveDecision:
        """Chooses next H using online dynamics predictions and uncertainty."""
        # 1. Cold start / identifiability check
        if self.last_observed_metric is None or not self.is_identified:
            self.fallback_count += 1
            # Fall back to V1 fixed-exponent model anchored at last observation
            if self.last_observed_metric is not None and self.last_observed_steps is not None:
                obs_m = self.last_observed_metric
                obs_s = self.last_observed_steps
                cand_preds = {
                    h: obs_m * ((float(h) / float(obs_s)) ** self.fallback_exponent)
                    for h in self.candidates
                }
                cand_costs = self.cost_estimator.predict_all(self.candidates, steps_remaining_in_epoch)
                feasible = [h for h in self.candidates if cand_preds[h] <= self.budget]
                if feasible:
                    target_h = min(feasible, key=lambda h: (cand_costs[h], -h))
                    fallback_reason = "IDENTIFIABILITY_FALLBACK_V1_FEASIBLE"
                else:
                    target_h = self.candidates[0]
                    fallback_reason = "IDENTIFIABILITY_FALLBACK_V1_MIN"
            else:
                target_h = self.current_h
                cand_preds = {}
                cand_costs = {}
                feasible = []
                fallback_reason = "COLD_START_KEEP"

            next_h = self.clamp_transition(target_h)
            dec_label = "INCREASE" if next_h > self.current_h else ("DECREASE" if next_h < self.current_h else "KEEP")

            metadata = {
                "controller_type": "ODD-H (V2)",
                "identified": False,
                "distinct_h_count": len(self.observed_h_levels),
                "total_observations": self.total_observations,
                "learned_exponent_p": float(self.theta[1]),
                "fallback_reason": fallback_reason,
                "candidate_predictions": cand_preds,
                "candidate_costs": cand_costs,
                "feasible_candidates": feasible,
            }
            return AdaptiveDecision(
                next_h=next_h,
                decision=dec_label,
                reason=(
                    f"Predictor not ready (observed H levels: {sorted(self.observed_h_levels)}, "
                    f"obs={self.total_observations}). Fallback to V1 logic: {fallback_reason}."
                ),
                metadata=metadata,
            )

        # 2. Identified model prediction
        cand_preds = {}
        cand_uncertainties = {}
        cand_uppers = {}
        for h in self.candidates:
            pred, unc, upper = self.predict_candidate(h)
            cand_preds[h] = pred
            cand_uncertainties[h] = unc
            cand_uppers[h] = upper

        cand_costs = self.cost_estimator.predict_all(self.candidates, steps_remaining_in_epoch)

        # 3. Determine feasibility under conservative upper prediction U_t(H) <= Budget
        feasible = [h for h in self.candidates if cand_uppers[h] <= self.budget]

        if not feasible:
            self.budget_violation_count += 1
            target_h = self.candidates[0]
            reason = (
                f"Budget violation under uncertainty: no candidate feasible under budget {self.budget:.4e}. "
                f"Min upper prediction: {min(cand_uppers.values()):.4e}. Falling back to H_min={target_h}."
            )
            raw_decision = "FALLBACK_MIN"
        else:
            target_h = min(feasible, key=lambda h: (cand_costs[h], -h))
            reason = (
                f"Selected H={target_h} from feasible {feasible} with upper prediction={cand_uppers[target_h]:.4e} "
                f"(nominal={cand_preds[target_h]:.4e}, uncertainty={cand_uncertainties[target_h]:.4e}) <= budget {self.budget:.4e}."
            )
            raw_decision = "SELECT_FEASIBLE"

        next_h = self.clamp_transition(target_h)
        dec_label = "INCREASE" if next_h > self.current_h else ("DECREASE" if next_h < self.current_h else "KEEP")

        metadata = {
            "controller_type": "ODD-H (V2)",
            "identified": True,
            "distinct_h_count": len(self.observed_h_levels),
            "learned_exponent_p": float(self.theta[1]),
            "learned_log_a_b": float(self.theta[0]),
            "candidate_nominal_predictions": cand_preds,
            "candidate_uncertainties": cand_uncertainties,
            "candidate_upper_predictions": cand_uppers,
            "candidate_costs": cand_costs,
            "feasible_candidates": feasible,
            "raw_decision": raw_decision,
            "clipping_events": self.clipping_events,
            "last_residual": self.last_residual,
        }

        return AdaptiveDecision(
            next_h=next_h,
            decision=dec_label,
            reason=reason,
            metadata=metadata,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "current_h": self.current_h,
            "sync_rounds": self.sync_rounds,
            "budget": self.budget,
            "metric_type": self.metric_type,
            "fallback_exponent": self.fallback_exponent,
            "lambda_rls": self.lambda_rls,
            "kappa": self.kappa,
            "min_distinct_h": self.min_distinct_h,
            "min_observations": self.min_observations,
            "theta": self.theta.tolist(),
            "P": self.P.tolist(),
            "residual_var": self.residual_var,
            "last_residual": self.last_residual,
            "observed_h_levels": list(self.observed_h_levels),
            "total_observations": self.total_observations,
            "clipping_events": self.clipping_events,
            "fallback_count": self.fallback_count,
            "budget_violation_count": self.budget_violation_count,
            "last_observed_metric": self.last_observed_metric,
            "last_observed_steps": self.last_observed_steps,
            "cost_estimator": self.cost_estimator.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.current_h = int(state.get("current_h", self.current_h))
        self.sync_rounds = int(state.get("sync_rounds", self.sync_rounds))
        self.budget = float(state.get("budget", self.budget))
        self.metric_type = str(state.get("metric_type", self.metric_type))
        self.fallback_exponent = float(state.get("fallback_exponent", self.fallback_exponent))
        self.lambda_rls = float(state.get("lambda_rls", self.lambda_rls))
        self.kappa = float(state.get("kappa", self.kappa))
        self.min_distinct_h = int(state.get("min_distinct_h", self.min_distinct_h))
        self.min_observations = int(state.get("min_observations", self.min_observations))
        if "theta" in state:
            self.theta = np.array(state["theta"], dtype=np.float64)
        if "P" in state:
            self.P = np.array(state["P"], dtype=np.float64)
        self.residual_var = float(state.get("residual_var", self.residual_var))
        self.last_residual = float(state.get("last_residual", self.last_residual))
        if "observed_h_levels" in state:
            self.observed_h_levels = set(state["observed_h_levels"])
        self.total_observations = int(state.get("total_observations", self.total_observations))
        self.clipping_events = int(state.get("clipping_events", self.clipping_events))
        self.fallback_count = int(state.get("fallback_count", self.fallback_count))
        self.budget_violation_count = int(state.get("budget_violation_count", self.budget_violation_count))
        self.last_observed_metric = state.get("last_observed_metric", self.last_observed_metric)
        self.last_observed_steps = state.get("last_observed_steps", self.last_observed_steps)
        if "cost_estimator" in state:
            self.cost_estimator.load_state_dict(state["cost_estimator"])
