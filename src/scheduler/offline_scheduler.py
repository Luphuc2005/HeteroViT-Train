"""Offline Scheduler for Distributed Heterogeneous Workload Evaluation and Validation."""
import os
import sys
import json
import csv
import argparse
from typing import Dict, Any, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.models.compute_model import ComputeCostModel
from src.scheduler.models.communication_model import CommunicationCostModel
from src.scheduler.models.straggler_model import StragglerPenaltyModel
from src.scheduler.cost_model import CostModelEvaluator, CandidateEvaluation
from src.scheduler.candidate_generator import CandidateGenerator


class OfflineScheduler:
    """Orchestrates candidate evaluation, ranking, and offline validation."""

    def __init__(
        self,
        compute_profile_path: str = os.path.join(PROJECT_ROOT, "profiles", "compute_profile.json"),
        network_profile_path: str = os.path.join(PROJECT_ROOT, "profiles", "network_profile.json"),
        runtime_profile_path: Optional[str] = os.path.join(PROJECT_ROOT, "profiles", "runtime_profile.json"),
        lambda_penalty: float = 1.0,
    ):
        self.compute_model = ComputeCostModel(compute_profile_path)
        self.comm_model = CommunicationCostModel(network_profile_path)
        self.straggler_model = StragglerPenaltyModel(
            lambda_penalty=lambda_penalty,
            runtime_profile_path_or_dict=runtime_profile_path if (runtime_profile_path and os.path.exists(runtime_profile_path)) else None,
        )
        self.evaluator = CostModelEvaluator(
            compute_model=self.compute_model,
            communication_model=self.comm_model,
            straggler_model=self.straggler_model,
            lambda_penalty=lambda_penalty,
        )

    def evaluate(self, candidate: Dict[str, int]) -> CandidateEvaluation:
        return self.evaluator.evaluate(candidate)

    def rank_candidates(self, candidates: List[Dict[str, int]]) -> List[CandidateEvaluation]:
        return self.evaluator.rank_candidates(candidates)

    def print_ranking_table(self, evaluations: List[CandidateEvaluation]) -> None:
        """Prints formatted ranking table of candidates."""
        print("=" * 115)
        print(f"{'Rank':<5} | {'Candidate Allocation':<36} | {'Nodes':<5} | {'Batch':<6} | {'T_crit (ms)':<12} | {'Bottleneck':<10} | {'Pred Tput':<12}")
        print("-" * 115)
        for idx, ev in enumerate(evaluations, start=1):
            alloc_str = ", ".join(f"{k}:{v}" for k, v in ev.candidate.items())
            num_nodes = len(ev.active_nodes)
            print(
                f"{idx:<5} | {alloc_str:<36} | {num_nodes:<5} | {ev.total_batch:<6} | "
                f"{ev.t_critical_ms:10.2f} ms | {ev.critical_node:<10} | {ev.predicted_throughput:8.1f} img/s"
            )
        print("=" * 115)

    def print_detailed_breakdown(self, ev: CandidateEvaluation, label: str = "") -> None:
        """Prints per-node breakdown for an evaluated candidate."""
        header = f" CANDIDATE BREAKDOWN: {label} (Total Batch: {ev.total_batch}, Pred Tput: {ev.predicted_throughput:.1f} img/s) "
        print("\n" + "=" * 95)
        print(f"{header:^95}")
        print("=" * 95)
        print(f"{'Node':<8} | {'Batch':<6} | {'Status':<8} | {'Compute (ms)':<14} | {'Comm (ms)':<11} | {'Straggler (ms)':<14} | {'Total Cost (ms)':<15}")
        print("-" * 95)
        for node, d in ev.node_breakdown.items():
            status = "ACTIVE" if d["active"] else "INACTIVE"
            crit_marker = " <-- CRITICAL PATH" if node == ev.critical_node and d["active"] else ""
            print(
                f"{node:<8} | {d['batch']:<6} | {status:<8} | "
                f"{d['compute_ms']:12.2f} ms | {d['comm_ms']:9.2f} ms | "
                f"{d['penalty_ms']:12.2f} ms | {d['total_cost_ms']:13.2f} ms{crit_marker}"
            )
        print("-" * 95)
        print(f"Critical Path Time T_critical: {ev.t_critical_ms:.2f} ms (Node: {ev.critical_node})")
        print(f"Predicted Throughput         : {ev.predicted_throughput:.2f} images/second")
        print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Offline Heterogeneous Scheduler Evaluator")
    parser.add_argument("--lambda-penalty", type=float, default=1.0, help="Straggler penalty multiplier lambda")
    parser.add_argument("--compute-profile", type=str, default=os.path.join(PROJECT_ROOT, "profiles", "compute_profile.json"))
    parser.add_argument("--network-profile", type=str, default=os.path.join(PROJECT_ROOT, "profiles", "network_profile.json"))
    parser.add_argument("--runtime-profile", type=str, default=os.path.join(PROJECT_ROOT, "profiles", "runtime_profile.json"))
    args = parser.parse_args()

    scheduler = OfflineScheduler(
        compute_profile_path=args.compute_profile,
        network_profile_path=args.network_profile,
        runtime_profile_path=args.runtime_profile,
        lambda_penalty=args.lambda_penalty,
    )

    candidates = CandidateGenerator.get_benchmark_candidates()
    ranked = scheduler.rank_candidates(candidates)

    print("\n>>> Phase 1 Candidate Evaluation & Ranking:")
    scheduler.print_ranking_table(ranked)

    for idx, ev in enumerate(ranked, start=1):
        scheduler.print_detailed_breakdown(ev, label=f"Rank {idx}")


if __name__ == "__main__":
    main()
