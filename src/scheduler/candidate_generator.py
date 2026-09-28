"""Candidate Generator for 5-Node Heterogeneous Training Allocations."""
from typing import Dict, List, Optional


class CandidateGenerator:
    """Generates candidate workload allocations (5-node and 4-node without stragglers)."""

    @staticmethod
    def get_benchmark_candidates() -> List[Dict[str, int]]:
        """Returns representative candidates for Phase 1 validation comparison."""
        return [
            # C1: Uniform 5-node Baseline (Equal local batch 128, Global 640)
            {
                "lab01": 128,
                "lab02": 128,
                "lab03": 128,
                "lab04": 128,
                "lab05": 128,
            },
            # C2: Heterogeneous Static 5-node (Previously configured in hetero_static_5nodes.yaml)
            {
                "lab01": 384,
                "lab02": 64,
                "lab03": 48,
                "lab04": 48,
                "lab05": 96,
            },
            # C3: Balanced 5-node (Workload proportional to empirical compute capacities)
            {
                "lab01": 256,
                "lab02": 16,
                "lab03": 24,
                "lab04": 24,
                "lab05": 48,
            },
            # C4: 4-node Candidate (Drop slowest CPU straggler lab02 completely: batch=0)
            {
                "lab01": 256,
                "lab02": 0,
                "lab03": 32,
                "lab04": 32,
                "lab05": 64,
            },
            # C5: 4-node Candidate (Drop Core i3 lab03 completely: batch=0)
            {
                "lab01": 256,
                "lab02": 24,
                "lab03": 0,
                "lab04": 32,
                "lab05": 64,
            },
        ]
