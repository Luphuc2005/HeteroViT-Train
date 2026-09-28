"""Cost Model and Candidate Evaluator for Distributed Heterogeneous Training."""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from src.scheduler.models.compute_model import ComputeCostModel
from src.scheduler.models.communication_model import CommunicationCostModel
from src.scheduler.models.straggler_model import StragglerPenaltyModel


@dataclass
class CandidateEvaluation:
    """Detailed evaluation result for a candidate allocation."""
    candidate: Dict[str, int]
    active_nodes: List[str]
    inactive_nodes: List[str]
    total_batch: int
    node_breakdown: Dict[str, Dict[str, float]]
    critical_node: str
    t_critical_ms: float
    predicted_throughput: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate,
            "active_nodes": self.active_nodes,
            "inactive_nodes": self.inactive_nodes,
            "total_batch": self.total_batch,
            "node_breakdown": self.node_breakdown,
            "critical_node": self.critical_node,
            "t_critical_ms": self.t_critical_ms,
            "predicted_throughput": self.predicted_throughput,
        }


class CostModelEvaluator:
    """Evaluates candidates using C_i(b_i) = T_compute,i(b_i) + T_comm,i + P_straggler,i."""

    def __init__(
        self,
        compute_model: ComputeCostModel,
        communication_model: CommunicationCostModel,
        straggler_model: StragglerPenaltyModel,
        lambda_penalty: float = 1.0,
    ):
        self.compute_model = compute_model
        self.comm_model = communication_model
        self.straggler_model = straggler_model
        self.lambda_penalty = float(lambda_penalty)
        self.straggler_model.lambda_penalty = self.lambda_penalty

    def evaluate(self, candidate: Dict[str, int]) -> CandidateEvaluation:
        """Evaluates an allocation candidate dict e.g. {'lab01': 256, 'lab02': 0, ...}."""
        active_nodes = [node for node, b in candidate.items() if b > 0]
        inactive_nodes = [node for node, b in candidate.items() if b <= 0]
        total_batch = sum(max(0, int(b)) for b in candidate.values())

        if not active_nodes or total_batch == 0:
            raise ValueError("Candidate must have at least one active node with batch > 0")

        # Communication cost is shared across all active nodes in the AllReduce communicator
        t_comm_ms = self.comm_model.predict_comm_ms(active_nodes)

        node_breakdown = {}
        max_cost_ms = -1.0
        critical_node = active_nodes[0]

        for node, b in candidate.items():
            b = int(b)
            if b <= 0:
                node_breakdown[node] = {
                    "batch": 0,
                    "active": False,
                    "compute_ms": 0.0,
                    "comm_ms": 0.0,
                    "penalty_ms": 0.0,
                    "total_cost_ms": 0.0,
                }
                continue

            t_compute = self.compute_model.predict_compute_ms(node, b)
            compute_std = self.compute_model.get_compute_std_ms(node, b)
            penalty = self.straggler_model.compute_penalty_ms(node, fallback_std_ms=compute_std)
            total_node_cost = t_compute + t_comm_ms + penalty

            node_breakdown[node] = {
                "batch": b,
                "active": True,
                "compute_ms": t_compute,
                "comm_ms": t_comm_ms,
                "penalty_ms": penalty,
                "total_cost_ms": total_node_cost,
            }

            if total_node_cost > max_cost_ms:
                max_cost_ms = total_node_cost
                critical_node = node

        t_critical_ms = max_cost_ms
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
        """Evaluates multiple candidates and sorts them descending by predicted throughput."""
        evaluations = [self.evaluate(c) for c in candidates]
        evaluations.sort(key=lambda x: x.predicted_throughput, reverse=True)
        return evaluations
