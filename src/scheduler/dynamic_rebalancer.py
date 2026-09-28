"""Dynamic Load Balancer with Hysteresis and Cooldown for Heterogeneous Training."""
import os
import csv
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

from src.scheduler.online_cost_model import OnlineCostModel
from src.scheduler.cost_model import CandidateEvaluation
from src.scheduler.candidate_generator import CandidateGenerator


@dataclass
class RebalanceDecision:
    """Auditable record of a load balancing decision at an epoch boundary."""
    epoch: int
    current_allocation: Dict[str, int]
    target_allocation: Dict[str, int]
    action: str  # "KEEP" or "SWITCH"
    reason: str
    predicted_gain_pct: float
    current_t_critical_ms: float
    target_t_critical_ms: float
    current_throughput: float
    target_throughput: float
    cooldown_remaining: int
    scheduler_overhead_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DynamicRebalancer:
    """Master-side closed-loop dynamic workload rebalancer.
    
    Observes actual per-worker runtime telemetry, calibrates the online Cost Model,
    evaluates reallocation candidates, and applies hysteresis/cooldown guards.
    """

    def __init__(
        self,
        cost_model: OnlineCostModel,
        epsilon: float = 0.05,
        cooldown_epochs: int = 1,
        slowdown_threshold_r: float = 1.15,
        log_csv_path: Optional[str] = None,
        **kwargs,
    ):
        self.cost_model = cost_model
        self.epsilon = float(epsilon)
        self.cooldown_epochs = max(0, int(cooldown_epochs))
        self.slowdown_threshold_r = float(slowdown_threshold_r)
        self.log_csv_path = log_csv_path
        self.last_switch_epoch: Optional[int] = None
        self.switch_count = 0
        self.decision_count = 0

        if self.log_csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_csv_path)), exist_ok=True)
            if not os.path.exists(self.log_csv_path):
                with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch",
                        "current_allocation",
                        "target_allocation",
                        "active_nodes",
                        "per_node_compute_ema",
                        "runtime_std",
                        "comm_time_ms",
                        "idle_time_ms",
                        "predicted_t_critical_ms",
                        "measured_t_critical_ms",
                        "predicted_throughput",
                        "scheduler_decision",
                        "predicted_gain_pct",
                        "reason",
                        "scheduler_overhead_ms",
                    ])

    def cooldown_remaining(self, epoch: int) -> int:
        if self.last_switch_epoch is None:
            return 0
        elapsed = int(epoch) - self.last_switch_epoch
        if elapsed <= self.cooldown_epochs:
            return max(0, self.cooldown_epochs - elapsed + 1)
        return 0

    def decide_rebalance(
        self,
        epoch: int,
        telemetry_list: List[Dict[str, Any]],
        current_allocation: Dict[str, int],
        global_batch: int,
        measured_t_critical_ms: Optional[float] = None,
    ) -> RebalanceDecision:
        """Processes worker telemetry, ranks candidates, and decides whether to rebalance."""
        t_start = time.perf_counter()

        # 1. Update online states with incoming worker telemetry
        for t in telemetry_list:
            node_id = t["node_id"]
            compute_ms = float(t.get("compute_ms", 0.0))
            comm_ms = float(t.get("comm_ms", 0.0))
            idle_ms = float(t.get("idle_ms", 0.0))
            batch = int(t.get("current_batch", current_allocation.get(node_id, 0)))
            std_ms = float(t.get("runtime_std_ms", 0.0))
            rank = int(t.get("rank", 0))

            self.cost_model.cluster_state.update_node(
                node_id=node_id,
                compute_ms=compute_ms,
                comm_ms=comm_ms,
                idle_ms=idle_ms,
                current_batch=batch,
                runtime_std_ms=std_ms,
                rank=rank,
            )

        # 2. Check for slowdown anomalies or recovery among active nodes
        slow_nodes = []
        recovered_nodes = []
        for node_id, b in current_allocation.items():
            if b > 0:
                r_factor = self.cost_model.compute_correction_factor(node_id)
                if r_factor >= self.slowdown_threshold_r:
                    slow_nodes.append((node_id, r_factor))
                nominal_b = 48 if node_id in ("lab03", "lab04") else (64 if node_id == "lab02" else 96)
                if b < nominal_b and r_factor <= 1.05:
                    recovered_nodes.append((node_id, r_factor))

        # Sort slow nodes by severity (highest r first)
        slow_nodes.sort(key=lambda x: x[1], reverse=True)
        primary_slow_node = slow_nodes[0][0] if slow_nodes else None

        # 3. Evaluate current allocation baseline
        curr_eval = self.cost_model.evaluate_candidate(current_allocation)

        # 4. Check whether to evaluate reallocation:
        # If in cooldown window, hold current allocation
        remaining_cooldown = self.cooldown_remaining(epoch)

        if remaining_cooldown > 0:
            action = "KEEP"
            reason = f"cooldown_active ({remaining_cooldown} epochs remaining)"
            best_eval = curr_eval
            gain_pct = 0.0
        elif not slow_nodes and not recovered_nodes:
            # Cluster is operating stably within expected empirical profiles
            action = "KEEP"
            reason = "cluster_stable_nominal_performance"
            best_eval = curr_eval
            gain_pct = 0.0
        else:
            # Generate valid candidates matching global_batch
            candidates = CandidateGenerator.generate_candidates(
                global_batch_size=global_batch,
                current_allocation=current_allocation,
                slow_node_id=primary_slow_node,
            )

            # Evaluate and rank all candidate allocations
            ranked_evals = self.cost_model.rank_candidates(candidates)
            best_eval = ranked_evals[0]

            is_same = (best_eval.candidate == current_allocation)
            gain_pct = 0.0
            if curr_eval.t_critical_ms > 0:
                gain = (curr_eval.t_critical_ms - best_eval.t_critical_ms) / curr_eval.t_critical_ms
                gain_pct = float(gain * 100.0)

            if is_same:
                action = "KEEP"
                reason = "current_allocation_is_optimal"
            elif (gain_pct / 100.0) <= self.epsilon:
                action = "KEEP"
                reason = f"predicted_gain_{gain_pct:.1f}%_below_epsilon_{self.epsilon*100:.1f}%"
            else:
                action = "SWITCH"
                if primary_slow_node:
                    r_val = slow_nodes[0][1]
                    node_state = self.cost_model.cluster_state.get_node(primary_slow_node)
                    ema_val = node_state.compute_ema_ms if node_state else 0.0
                    reason = f"{primary_slow_node}_slowdown_detected (r={r_val:.2f}, EMA={ema_val:.0f}ms, gain={gain_pct:.1f}%)"
                elif recovered_nodes:
                    rec_node = recovered_nodes[0][0]
                    reason = f"{rec_node}_recovery_detected (workload_restored, gain={gain_pct:.1f}%)"
                else:
                    reason = f"throughput_optimization_rebalance (gain={gain_pct:.1f}%)"
                self.last_switch_epoch = int(epoch)
                self.switch_count += 1

        self.decision_count += 1
        overhead_ms = (time.perf_counter() - t_start) * 1000.0

        decision = RebalanceDecision(
            epoch=int(epoch),
            current_allocation=dict(current_allocation),
            target_allocation=dict(best_eval.candidate),
            action=action,
            reason=reason,
            predicted_gain_pct=gain_pct,
            current_t_critical_ms=curr_eval.t_critical_ms,
            target_t_critical_ms=best_eval.t_critical_ms,
            current_throughput=curr_eval.predicted_throughput,
            target_throughput=best_eval.predicted_throughput,
            cooldown_remaining=remaining_cooldown,
            scheduler_overhead_ms=overhead_ms,
        )

        # 5. Audit Logging to CSV
        if self.log_csv_path:
            active_nodes = [n for n, b in current_allocation.items() if b > 0]
            per_node_ema = {
                n: round(self.cost_model.cluster_state.get_node(n).compute_ema_ms, 2)
                for n in current_allocation
                if self.cost_model.cluster_state.get_node(n)
            }
            per_node_std = {
                n: round(self.cost_model.cluster_state.get_node(n).runtime_std_ms, 2)
                for n in current_allocation
                if self.cost_model.cluster_state.get_node(n)
            }
            avg_comm = float(curr_eval.node_breakdown.get(active_nodes[0], {}).get("comm_ms", 0.0))
            avg_idle = float(sum(t.get("idle_ms", 0.0) for t in telemetry_list) / max(len(telemetry_list), 1))
            meas_crit = float(measured_t_critical_ms) if measured_t_critical_ms is not None else curr_eval.t_critical_ms

            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    str(current_allocation),
                    str(best_eval.candidate),
                    str(active_nodes),
                    str(per_node_ema),
                    str(per_node_std),
                    f"{avg_comm:.2f}",
                    f"{avg_idle:.2f}",
                    f"{curr_eval.t_critical_ms:.2f}",
                    f"{meas_crit:.2f}",
                    f"{curr_eval.predicted_throughput:.2f}",
                    action,
                    f"{gain_pct:.2f}",
                    reason,
                    f"{overhead_ms:.3f}",
                ])

        return decision
