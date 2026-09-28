"""Online Cost Model with Empirical Profile Calibration and Dynamic Telemetry Correction."""
import os
from typing import Dict, Any, List, Optional
from src.scheduler.models.compute_model import ComputeCostModel
from src.scheduler.models.communication_model import CommunicationCostModel
from src.scheduler.cost_model import CandidateEvaluation
from src.scheduler.online_state import OnlineClusterState


class OnlineCostModel:
    """Predicts execution costs using empirical profiles adjusted by online EMA correction factors."""

    def __init__(
        self,
        compute_model: ComputeCostModel,
        communication_model: CommunicationCostModel,
        cluster_state: OnlineClusterState,
        r_min: float = 0.5,
        r_max: float = 3.0,
        lambda_penalty: float = 1.0,
    ):
        self.compute_model = compute_model
        self.comm_model = communication_model
        self.cluster_state = cluster_state
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.lambda_penalty = float(lambda_penalty)

    def compute_correction_factor(self, node_id: str) -> float:
        """Calculates clamped online correction factor r_i = EMA_compute_i / T_profile_i(current_batch).
        
        Returns 1.0 if the node has no prior valid observations or is inactive.
        """
        node_state = self.cluster_state.get_node(node_id)
        if node_state is None or node_state.current_batch <= 0 or node_state.compute_ema_ms <= 0.0:
            return 1.0

        try:
            t_profile = self.compute_model.predict_compute_ms(node_id, node_state.current_batch)
        except Exception:
            return 1.0

        if t_profile <= 0.0:
            return 1.0

        raw_r = node_state.compute_ema_ms / t_profile
        clamped_r = max(self.r_min, min(self.r_max, raw_r))
        return float(clamped_r)

    def predict_node_compute_ms(self, node_id: str, candidate_batch: int) -> float:
        """Predicts compute time: T_compute_i(batch, t) = r_i * T_profile_i(batch)."""
        candidate_batch = int(candidate_batch)
        if candidate_batch <= 0:
            return 0.0

        r_i = self.compute_correction_factor(node_id)
        t_prof = self.compute_model.predict_compute_ms(node_id, candidate_batch)
        return float(r_i * t_prof)

    def predict_uncertainty_penalty_ms(self, node_id: str, candidate_batch: int) -> float:
        """Calculates uncertainty penalty = lambda * runtime_std_i."""
        if candidate_batch <= 0:
            return 0.0

        node_state = self.cluster_state.get_node(node_id)
        if node_state is not None and node_state.runtime_std_ms > 0.0:
            std_ms = node_state.runtime_std_ms
        else:
            std_ms = self.compute_model.get_compute_std_ms(node_id, candidate_batch)

        return float(self.lambda_penalty * max(0.0, std_ms))

    def predict_comm_ms(self, active_nodes: List[str]) -> float:
        """Predicts AllReduce communication cost based on active node count."""
        return self.comm_model.predict_comm_ms(active_nodes)

    def evaluate_candidate(self, candidate: Dict[str, int]) -> CandidateEvaluation:
        """Evaluates candidate allocation using:
        
        T_critical = max_{i in active} ( T_compute_i(b_i) + penalty_i ) + T_comm(active_nodes)
        Throughput = total_batch / (T_critical / 1000.0)
        """
        active_nodes = [node for node, b in candidate.items() if int(b) > 0]
        inactive_nodes = [node for node, b in candidate.items() if int(b) <= 0]
        total_batch = sum(max(0, int(b)) for b in candidate.values())

        if not active_nodes or total_batch == 0:
            raise ValueError("Candidate must have at least one active node with batch > 0")

        t_comm_ms = self.predict_comm_ms(active_nodes)

        node_breakdown = {}
        max_node_time_ms = -1.0
        critical_node = active_nodes[0]

        for node, b in candidate.items():
            b = int(b)
            if b <= 0:
                node_breakdown[node] = {
                    "batch": 0,
                    "active": False,
                    "correction_factor": 1.0,
                    "compute_ms": 0.0,
                    "comm_ms": 0.0,
                    "penalty_ms": 0.0,
                    "total_node_time_ms": 0.0,
                }
                continue

            r_i = self.compute_correction_factor(node)
            t_comp = self.predict_node_compute_ms(node, b)
            penalty = self.predict_uncertainty_penalty_ms(node, b)
            total_node_time = t_comp + penalty

            node_breakdown[node] = {
                "batch": b,
                "active": True,
                "correction_factor": r_i,
                "compute_ms": t_comp,
                "comm_ms": t_comm_ms,
                "penalty_ms": penalty,
                "total_node_time_ms": total_node_time,
            }

            if total_node_time > max_node_time_ms:
                max_node_time_ms = total_node_time
                critical_node = node

        t_critical_ms = max_node_time_ms + t_comm_ms
        t_critical_s = t_critical_ms / 1000.0
        predicted_throughput = float(total_batch) / max(t_critical_s, 1e-6)

        return CandidateEvaluation(
            candidate=dict(candidate),
            active_nodes=active_nodes,
            inactive_nodes=inactive_nodes,
            total_batch=total_batch,
            node_breakdown=node_breakdown,
            critical_node=critical_node,
            t_critical_ms=t_critical_ms,
            predicted_throughput=predicted_throughput,
        )

    def rank_candidates(self, candidates: List[Dict[str, int]]) -> List[CandidateEvaluation]:
        evaluations = [self.evaluate_candidate(c) for c in candidates]
        evaluations.sort(key=lambda x: x.t_critical_ms)
        return evaluations
