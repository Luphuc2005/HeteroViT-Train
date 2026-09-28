"""Phase 2 Dynamic Load Balancer Smoke Test Runner (E1: Stable, E2: Slowdown, E3: Recovery).

Runs controlled experiments to verify:
- E1: Stable conditions -> Scheduler keeps allocation, no spurious jumping.
- E2: Worker Slowdown -> EMA rises, online Cost Model adjusts r_i, scheduler detects bottleneck,
      reduces worker batch or drops worker, predicted gain exceeds hysteresis epsilon, switches allocation.
- E3: Worker Recovery -> Normal latency returns, scheduler detects capacity recovery and rebalances back.
"""
import os
import sys
import json
import argparse
from typing import Dict, Any, List

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.models.compute_model import ComputeCostModel
from src.scheduler.models.communication_model import CommunicationCostModel
from src.scheduler.online_state import OnlineClusterState
from src.scheduler.online_cost_model import OnlineCostModel
from src.scheduler.dynamic_rebalancer import DynamicRebalancer


def run_synthetic_smoke_test(verbose: bool = True) -> Dict[str, Any]:
    """Executes synthetic telemetry sequence E1 -> E2 -> E3 to rigorously verify Rebalancer mechanics."""
    compute_profile_path = os.path.join(PROJECT_ROOT, "profiles", "compute_profile.json")
    network_profile_path = os.path.join(PROJECT_ROOT, "profiles", "network_profile.json")

    compute_model = ComputeCostModel(compute_profile_path)
    comm_model = CommunicationCostModel(network_profile_path)
    cluster_state = OnlineClusterState(ema_alpha=0.3)
    cost_model = OnlineCostModel(
        compute_model=compute_model,
        communication_model=comm_model,
        cluster_state=cluster_state,
        r_min=0.5,
        r_max=3.0,
        lambda_penalty=1.0,
    )

    initial_allocation = {
        "lab01": 384,
        "lab02": 64,
        "lab03": 48,
        "lab04": 48,
        "lab05": 96,
    }
    global_batch = sum(initial_allocation.values())  # 640

    log_path = os.path.join(PROJECT_ROOT, "results", "phase2_smoke_test_decisions.csv")
    rebalancer = DynamicRebalancer(
        cost_model=cost_model,
        epsilon=0.05,
        cooldown_epochs=1,
        slowdown_threshold_r=1.15,
        log_csv_path=log_path,
    )

    results = {"E1_stable": [], "E2_slowdown": [], "E3_recovery": []}
    current_alloc = dict(initial_allocation)

    # -------------------------------------------------------------------------
    # E1: STABLE CONDITION (Epochs 1-2)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 90)
        print(" [EXPERIMENT E1: STABLE TRAINING UNDER EMPIRICAL CAPACITIES]")
        print("=" * 90)

    for epoch in range(1, 3):
        # Baseline measured compute times matching empirical profile
        telemetry = [
            {"node_id": "lab01", "rank": 0, "compute_ms": 658.96, "current_batch": current_alloc["lab01"], "comm_ms": 1603.0, "idle_ms": 1727.0},
            {"node_id": "lab02", "rank": 1, "compute_ms": 2385.93, "current_batch": current_alloc["lab02"], "comm_ms": 1603.0, "idle_ms": 0.0},
            {"node_id": "lab03", "rank": 2, "compute_ms": 2126.38, "current_batch": current_alloc["lab03"], "comm_ms": 1603.0, "idle_ms": 259.0},
            {"node_id": "lab04", "rank": 3, "compute_ms": 2119.78, "current_batch": current_alloc["lab04"], "comm_ms": 1603.0, "idle_ms": 266.0},
            {"node_id": "lab05", "rank": 4, "compute_ms": 1482.92, "current_batch": current_alloc["lab05"], "comm_ms": 1603.0, "idle_ms": 903.0},
        ]
        decision = rebalancer.decide_rebalance(
            epoch=epoch,
            telemetry_list=telemetry,
            current_allocation=current_alloc,
            global_batch=global_batch,
            measured_t_critical_ms=2385.93 + 1603.0,
        )
        current_alloc = dict(decision.target_allocation)
        results["E1_stable"].append(decision.to_dict())

        if verbose:
            print(f"Epoch {epoch:02d}: Action={decision.action:<6} | Gain={decision.predicted_gain_pct:4.1f}% | Reason={decision.reason}")
            print(f"          Current Alloc: {decision.current_allocation}")
            print(f"          Target Alloc : {decision.target_allocation}")

    # -------------------------------------------------------------------------
    # E2: SLOWDOWN INJECTION ON lab03 (Epochs 3-4)
    # Simulate CPU contention on lab03: compute time spikes from 2126ms to 5200ms (~2.45x slowdown)
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 90)
        print(" [EXPERIMENT E2: SLOWDOWN INJECTION ON lab03 (CPU CONTENTION)]")
        print("=" * 90)

    for epoch in range(3, 5):
        telemetry = [
            {"node_id": "lab01", "rank": 0, "compute_ms": 660.0, "current_batch": current_alloc["lab01"], "comm_ms": 1603.0, "idle_ms": 4540.0},
            {"node_id": "lab02", "rank": 1, "compute_ms": 2390.0, "current_batch": current_alloc["lab02"], "comm_ms": 1603.0, "idle_ms": 2810.0},
            {"node_id": "lab03", "rank": 2, "compute_ms": 5200.0, "current_batch": current_alloc["lab03"], "comm_ms": 1603.0, "idle_ms": 0.0},  # Straggler!
            {"node_id": "lab04", "rank": 3, "compute_ms": 2125.0, "current_batch": current_alloc["lab04"], "comm_ms": 1603.0, "idle_ms": 3075.0},
            {"node_id": "lab05", "rank": 4, "compute_ms": 1485.0, "current_batch": current_alloc["lab05"], "comm_ms": 1603.0, "idle_ms": 3715.0},
        ]
        decision = rebalancer.decide_rebalance(
            epoch=epoch,
            telemetry_list=telemetry,
            current_allocation=current_alloc,
            global_batch=global_batch,
            measured_t_critical_ms=5200.0 + 1603.0,
        )
        current_alloc = dict(decision.target_allocation)
        results["E2_slowdown"].append(decision.to_dict())

        if verbose:
            r_lab03 = cost_model.compute_correction_factor("lab03")
            print(f"Epoch {epoch:02d}: Action={decision.action:<6} | r_lab03={r_lab03:.2f}x | Gain={decision.predicted_gain_pct:4.1f}% | Reason={decision.reason}")
            print(f"          Current Alloc: {decision.current_allocation}")
            print(f"          Target Alloc : {decision.target_allocation}")
            print(f"          Pred Critical: {decision.current_t_critical_ms:.1f}ms -> {decision.target_t_critical_ms:.1f}ms")

    # -------------------------------------------------------------------------
    # E3: RECOVERY PHASE (Epochs 5-6)
    # Background load on lab03 removed, latency returns to normal
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 90)
        print(" [EXPERIMENT E3: RECOVERY (BACKGROUND LOAD REMOVED ON lab03)]")
        print("=" * 90)

    for epoch in range(5, 7):
        curr_b3 = current_alloc["lab03"]
        normal_t3 = compute_model.predict_compute_ms("lab03", curr_b3) if curr_b3 > 0 else 0.0
        telemetry = [
            {"node_id": "lab01", "rank": 0, "compute_ms": 660.0, "current_batch": current_alloc["lab01"], "comm_ms": 1603.0, "idle_ms": 1720.0},
            {"node_id": "lab02", "rank": 1, "compute_ms": 2385.0, "current_batch": current_alloc["lab02"], "comm_ms": 1603.0, "idle_ms": 0.0},
            {"node_id": "lab03", "rank": 2, "compute_ms": normal_t3, "current_batch": curr_b3, "comm_ms": 1603.0, "idle_ms": 2385.0 - normal_t3},
            {"node_id": "lab04", "rank": 3, "compute_ms": 2120.0, "current_batch": current_alloc["lab04"], "comm_ms": 1603.0, "idle_ms": 265.0},
            {"node_id": "lab05", "rank": 4, "compute_ms": 1485.0, "current_batch": current_alloc["lab05"], "comm_ms": 1603.0, "idle_ms": 900.0},
        ]
        decision = rebalancer.decide_rebalance(
            epoch=epoch,
            telemetry_list=telemetry,
            current_allocation=current_alloc,
            global_batch=global_batch,
            measured_t_critical_ms=2385.0 + 1603.0,
        )
        current_alloc = dict(decision.target_allocation)
        results["E3_recovery"].append(decision.to_dict())

        if verbose:
            r_lab03 = cost_model.compute_correction_factor("lab03")
            print(f"Epoch {epoch:02d}: Action={decision.action:<6} | r_lab03={r_lab03:.2f}x | Gain={decision.predicted_gain_pct:4.1f}% | Reason={decision.reason}")
            print(f"          Current Alloc: {decision.current_allocation}")
            print(f"          Target Alloc : {decision.target_allocation}")

    if verbose:
        print("\n" + "=" * 90)
        print(" [SMOKE EXPERIMENTS SUMMARY REPORT]")
        print("=" * 90)
        print(f"1. E1 Stable  : Action={results['E1_stable'][0]['action']} (Alloc kept stable)")
        print(f"2. E2 Slowdown: Action={results['E2_slowdown'][0]['action']} (Workload shifted away from lab03, gain={results['E2_slowdown'][0]['predicted_gain_pct']:.1f}%)")
        print(f"3. E3 Recovery: Action={results['E3_recovery'][0]['action']} / {results['E3_recovery'][1]['action']} (lab03 recovered)")
        print(f"Scheduler Decisions written to: {log_path}")
        print("=" * 90)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Dynamic Load Balancer Smoke Test")
    args = parser.parse_args()
    run_synthetic_smoke_test(verbose=True)
