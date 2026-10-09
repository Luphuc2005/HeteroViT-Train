"""Adaptive Local SGD package for HeteroViT-MPI.

Provides:
  - Consensus metrics (V_t, Q_t, M_t) with mathematical identity checks and MPI aggregation.
  - Shared epoch-aware cost estimator.
  - Controller base interface and dataclasses.
  - V1: Consensus-Budget Adaptive-H (CBA-H).
  - V2: Online Drift-Dynamics Adaptive-H (ODD-H).
  - V3: Primal-Dual Cost-Aware Adaptive-H (PDCA-H).
  - Factory dispatch and structured CSV logging.
"""

from .consensus_metrics import (
    ConsensusMetrics,
    compute_centralized_consensus_metrics,
    compute_local_consensus_stats,
    reduce_consensus_metrics,
)
from .cost_estimator import CostEstimator, EpochAwareCostModel
from .controller_base import SyncRecord, AdaptiveHControllerBase
from .controller_logging import AdaptiveStructuredLogger

__all__ = [
    "ConsensusMetrics",
    "compute_centralized_consensus_metrics",
    "compute_local_consensus_stats",
    "reduce_consensus_metrics",
    "CostEstimator",
    "EpochAwareCostModel",
    "SyncRecord",
    "AdaptiveHControllerBase",
    "AdaptiveStructuredLogger",
]

