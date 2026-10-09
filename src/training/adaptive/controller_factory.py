"""Controller Factory for Adaptive Local SGD.

Dispatches configuration to:
  - Original heuristic: 'drift_adaptive_h' / 'heuristic_v1'
  - V1 Consensus-Budget: 'cba_h' / 'consensus_budget' / 'v1'
  - V2 Online Drift-Dynamics: 'odd_h' / 'online_dynamics' / 'v2'
  - V3 Primal-Dual Cost-Aware: 'pdca_h' / 'primal_dual' / 'v3'
  - Fixed Local SGD: 'fixed' (returns None)
"""

from typing import Dict, Any, Optional
from .v1_consensus_budget import ConsensusBudgetController
from .v2_online_dynamics import OnlineDriftDynamicsController
from .v3_primal_dual import PrimalDualCostAwareController
from .cost_estimator import CostEstimator


def build_adaptive_controller(
    config: Dict[str, Any],
    run_dir: Optional[str] = None,
    rank: int = 0,
    comm = None,
) -> Optional[Any]:
    """Instantiates the requested adaptive controller from configuration dictionary."""
    training_cfg = config.get("training", {})
    policy = str(training_cfg.get("local_sgd_policy", "fixed")).lower()
    adaptive_cfg = training_cfg.get("adaptive_h", {})

    if policy in ("fixed", "none") and not adaptive_cfg.get("enabled", False):
        return None

    controller_type = str(adaptive_cfg.get("type", policy)).lower()
    candidates = [int(c) for c in adaptive_cfg.get("candidates", [40, 80, 120, 150])]
    initial_h = int(adaptive_cfg.get("initial_h", candidates[0]))
    steps_per_epoch = int(training_cfg.get("steps_per_epoch", 150))
    adjacent_step_only = bool(adaptive_cfg.get("adjacent_step_only", True))
    epoch_aware_cost = bool(adaptive_cfg.get("epoch_aware_cost", True))

    cost_est = CostEstimator(
        steps_per_epoch=steps_per_epoch,
        ema_alpha=float(adaptive_cfg.get("cost_ema_alpha", 0.2)),
        epoch_aware_cost=epoch_aware_cost,
    )

    # 1. Legacy / Heuristic Drift-Adaptive-H (Baseline V1)
    if controller_type in ("drift_adaptive_h", "heuristic_v1", "legacy_heuristic", "drift_adaptive"):
        from src.scheduler.adaptive_h_controller import DriftAdaptiveHController
        return DriftAdaptiveHController(
            config=adaptive_cfg,
            run_dir=run_dir,
            rank=rank,
            comm=comm,
        )

    # 2. V1: Consensus-Budget Adaptive-H (CBA-H)
    elif controller_type in ("cba_h", "consensus_budget", "adaptive_v1", "v1"):
        budget = float(adaptive_cfg.get("budget", adaptive_cfg.get("consensus_budget", 0.05)))
        metric_type = str(adaptive_cfg.get("metric_type", "normalized_q")).lower()
        growth_p = float(adaptive_cfg.get("growth_exponent", adaptive_cfg.get("p", 2.0)))
        return ConsensusBudgetController(
            candidates=candidates,
            initial_h=initial_h,
            budget=budget,
            metric_type=metric_type,
            growth_exponent=growth_p,
            steps_per_epoch=steps_per_epoch,
            adjacent_step_only=adjacent_step_only,
            cost_estimator=cost_est,
            epoch_aware_cost=epoch_aware_cost,
        )

    # 3. V2: Online Drift-Dynamics Adaptive-H (ODD-H)
    elif controller_type in ("odd_h", "online_dynamics", "adaptive_v2", "v2"):
        budget = float(adaptive_cfg.get("budget", adaptive_cfg.get("consensus_budget", 0.05)))
        metric_type = str(adaptive_cfg.get("metric_type", "normalized_q")).lower()
        forgetting = float(adaptive_cfg.get("forgetting_factor", adaptive_cfg.get("lambda_rls", 0.98)))
        kappa = float(adaptive_cfg.get("uncertainty_multiplier", adaptive_cfg.get("kappa", 1.0)))
        min_distinct_h = int(adaptive_cfg.get("min_distinct_h", 2))
        min_observations = int(adaptive_cfg.get("min_observations", 3))
        fallback_p = float(adaptive_cfg.get("fallback_exponent", 2.0))
        p_bounds = tuple(adaptive_cfg.get("p_bounds", [0.1, 4.0]))

        return OnlineDriftDynamicsController(
            candidates=candidates,
            initial_h=initial_h,
            budget=budget,
            metric_type=metric_type,
            fallback_exponent=fallback_p,
            forgetting_factor=forgetting,
            uncertainty_multiplier=kappa,
            min_distinct_h=min_distinct_h,
            min_observations=min_observations,
            p_bounds=p_bounds,
            steps_per_epoch=steps_per_epoch,
            adjacent_step_only=adjacent_step_only,
            cost_estimator=cost_est,
            epoch_aware_cost=epoch_aware_cost,
        )

    # 4. V3: Primal-Dual Cost-Aware Adaptive-H (PDCA-H)
    elif controller_type in ("pdca_h", "primal_dual", "adaptive_v3", "v3"):
        risk_budget = float(adaptive_cfg.get("risk_budget", adaptive_cfg.get("budget", 0.03)))
        dual_step_size = float(adaptive_cfg.get("dual_step_size", adaptive_cfg.get("eta_mu", 1.0)))
        dual_initial = float(adaptive_cfg.get("dual_initial", adaptive_cfg.get("mu_0", 0.0)))
        dual_max = adaptive_cfg.get("dual_max", 50.0)
        dual_max = float(dual_max) if dual_max is not None else None
        metric_type = str(adaptive_cfg.get("metric_type", "normalized_q")).lower()
        weighting_mode = str(adaptive_cfg.get("weighting_mode", "step_weighted")).lower()
        forgetting = float(adaptive_cfg.get("forgetting_factor", 0.98))
        kappa = float(adaptive_cfg.get("uncertainty_multiplier", 1.0))
        min_distinct_h = int(adaptive_cfg.get("min_distinct_h", 2))
        min_observations = int(adaptive_cfg.get("min_observations", 3))
        fallback_p = float(adaptive_cfg.get("fallback_exponent", 2.0))

        return PrimalDualCostAwareController(
            candidates=candidates,
            initial_h=initial_h,
            risk_budget=risk_budget,
            dual_step_size=dual_step_size,
            dual_initial=dual_initial,
            dual_max=dual_max,
            metric_type=metric_type,
            weighting_mode=weighting_mode,
            steps_per_epoch=steps_per_epoch,
            adjacent_step_only=adjacent_step_only,
            cost_estimator=cost_est,
            epoch_aware_cost=epoch_aware_cost,
            forgetting_factor=forgetting,
            uncertainty_multiplier=kappa,
            min_distinct_h=min_distinct_h,
            min_observations=min_observations,
            fallback_exponent=fallback_p,
        )

    else:
        raise ValueError(
            f"Unsupported adaptive controller policy/type: '{controller_type}'. "
            f"Supported: 'fixed', 'drift_adaptive_h' (legacy), 'cba_h' (V1), 'odd_h' (V2), 'pdca_h' (V3)."
        )

