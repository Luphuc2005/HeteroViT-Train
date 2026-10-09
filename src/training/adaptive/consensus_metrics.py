"""Consensus Disagreement and Model Drift Metrics for Distributed Training.

Formulation:
  Let m workers have local parameter vectors w_i in R^d.
  Sample weights: alpha_i = n_i / sum_j n_j >= 0, sum_i alpha_i = 1.
  Weighted global consensus model: w_bar = sum_i alpha_i * w_i.

Metrics:
  1. Weighted Consensus Error:
       V_t = sum_i alpha_i * ||w_i - w_bar||_2^2
     Equivalent pairwise identity:
       V_t = (1/2) * sum_i sum_j alpha_i * alpha_j * ||w_i - w_j||_2^2
  2. Normalized Consensus Error:
       Q_t = V_t / (||w_bar||_2^2 + epsilon)
  3. Maximum Worker Drift:
       M_t = max_i (||w_i - w_bar||_2 / (||w_bar||_2 + epsilon))

All calculations use float64 precision for scalar accumulation and handle edge cases
(identical weights, zero samples, NaN/Inf detection).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class ConsensusMetrics:
    """Encapsulates consensus disagreement and drift metrics for a sync round."""
    v_t: float               # Weighted squared consensus error
    q_t: float               # Normalized consensus error
    m_t: float               # Maximum normalized worker drift
    norm_w_bar_sq: float     # ||w_bar||_2^2
    norm_w_bar: float        # ||w_bar||_2
    local_drift_norm: float  # Worker's own ||w_i - w_bar||_2 / (||w_bar||_2 + eps)
    local_weight: float      # Worker's own alpha_i
    legacy_drift: float      # sum_i alpha_i * (||w_i - w_bar||_2 / (||w_bar||_2 + eps))

    def to_dict(self) -> Dict[str, float]:
        return {
            "v_t": self.v_t,
            "q_t": self.q_t,
            "m_t": self.m_t,
            "norm_w_bar_sq": self.norm_w_bar_sq,
            "norm_w_bar": self.norm_w_bar,
            "local_drift_norm": self.local_drift_norm,
            "local_weight": self.local_weight,
            "legacy_drift": self.legacy_drift,
        }


def compute_local_consensus_stats(
    flat_w: np.ndarray,
    global_flat_w: np.ndarray,
    samples_since_sync: int,
    total_samples: int,
    epsilon: float = 1.0e-12,
) -> Dict[str, Any]:
    """Computes local consensus statistics on a single rank before averaging weights.
    
    Reuses the existing flattened parameter buffers without extra allocations.
    Calculations use float64 to ensure numerical stability.
    """
    if np.isnan(flat_w).any() or np.isnan(global_flat_w).any():
        raise ValueError("Encountered NaN in model parameters during consensus metric computation.")
    if np.isinf(flat_w).any() or np.isinf(global_flat_w).any():
        raise ValueError("Encountered Inf in model parameters during consensus metric computation.")

    # Convert to float64 for stable inner products and difference norms
    w_local_64 = flat_w.astype(np.float64, copy=False)
    w_global_64 = global_flat_w.astype(np.float64, copy=False)

    diff = w_local_64 - w_global_64
    diff_sq = float(np.dot(diff, diff))
    diff_l2 = float(np.sqrt(max(0.0, diff_sq)))

    norm_w_bar_sq = float(np.dot(w_global_64, w_global_64))
    norm_w_bar = float(np.sqrt(max(0.0, norm_w_bar_sq)))

    alpha_i = float(samples_since_sync) / float(total_samples) if total_samples > 0 else 0.0
    denom = norm_w_bar + epsilon
    local_norm_drift = diff_l2 / denom

    return {
        "diff_sq": diff_sq,
        "diff_l2": diff_l2,
        "norm_w_bar_sq": norm_w_bar_sq,
        "norm_w_bar": norm_w_bar,
        "alpha_i": alpha_i,
        "local_norm_drift": local_norm_drift,
        "weighted_diff_sq": alpha_i * diff_sq,
        "weighted_norm_drift": alpha_i * local_norm_drift,
        "denom_sq": norm_w_bar_sq + epsilon,
    }


def reduce_consensus_metrics(
    comm,
    local_stats: Dict[str, Any],
    epsilon: float = 1.0e-12,
    world_size: int = 1,
) -> ConsensusMetrics:
    """Aggregates local consensus statistics across ranks via scalar MPI collectives.
    
    Overhead:
      - 1 Allreduce of 2 float64 values (SUM): [sum_i alpha_i * ||w_i - w_bar||^2, sum_i alpha_i * d_i]
      - 1 Allreduce of 1 float64 value (MAX): [max_i d_i]
      Total communication: 24 bytes (NO model vectors transferred).
    """
    from mpi4py import MPI

    if comm is not None and world_size > 1:
        # Sum reduction: weighted squared difference and weighted normalized drift
        sum_buf = np.array([
            local_stats["weighted_diff_sq"],
            local_stats["weighted_norm_drift"],
        ], dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, sum_buf, op=MPI.SUM)
        v_t = max(0.0, float(sum_buf[0]))
        legacy_drift = float(sum_buf[1])

        # Max reduction: maximum normalized worker drift
        max_buf = np.array([local_stats["local_norm_drift"]], dtype=np.float64)
        comm.Allreduce(MPI.IN_PLACE, max_buf, op=MPI.MAX)
        m_t = float(max_buf[0])
    else:
        v_t = max(0.0, float(local_stats["weighted_diff_sq"]))
        legacy_drift = float(local_stats["weighted_norm_drift"])
        m_t = float(local_stats["local_norm_drift"])

    norm_w_bar_sq = local_stats["norm_w_bar_sq"]
    norm_w_bar = local_stats["norm_w_bar"]
    denom_sq = norm_w_bar_sq + epsilon
    q_t = v_t / denom_sq

    return ConsensusMetrics(
        v_t=v_t,
        q_t=q_t,
        m_t=m_t,
        norm_w_bar_sq=norm_w_bar_sq,
        norm_w_bar=norm_w_bar,
        local_drift_norm=local_stats["local_norm_drift"],
        local_weight=local_stats["alpha_i"],
        legacy_drift=legacy_drift,
    )


def compute_centralized_consensus_metrics(
    weights_list: List[np.ndarray],
    sample_counts: Optional[List[int]] = None,
    epsilon: float = 1.0e-12,
) -> Tuple[ConsensusMetrics, float]:
    """Computes consensus metrics in a centralized setting (for unit testing and verification).
    
    Also computes the pairwise consensus error identity:
        V_pairwise = (1/2) * sum_i sum_j alpha_i * alpha_j * ||w_i - w_j||_2^2
    and verifies that |V_t - V_pairwise| is zero within floating-point tolerance.
    
    Returns:
        (ConsensusMetrics, pairwise_discrepancy)
    """
    m = len(weights_list)
    if m == 0:
        raise ValueError("weights_list must not be empty.")

    if sample_counts is None:
        sample_counts = [1] * m

    total_samples = float(sum(sample_counts))
    if total_samples <= 0:
        alphas = np.array([1.0 / m] * m, dtype=np.float64)
    else:
        alphas = np.array([float(s) / total_samples for s in sample_counts], dtype=np.float64)

    # Convert all weights to 1D float64
    w_arrs = [w.ravel().astype(np.float64) for w in weights_list]

    # Weighted global model w_bar = sum_i alpha_i * w_i
    w_bar = np.zeros_like(w_arrs[0])
    for a_i, w_i in zip(alphas, w_arrs):
        w_bar += a_i * w_i

    norm_w_bar_sq = float(np.dot(w_bar, w_bar))
    norm_w_bar = float(np.sqrt(max(0.0, norm_w_bar_sq)))
    denom = norm_w_bar + epsilon
    denom_sq = norm_w_bar_sq + epsilon

    # 1. Direct weighted consensus error: V_t = sum_i alpha_i * ||w_i - w_bar||^2
    v_t = 0.0
    legacy_drift = 0.0
    m_t = 0.0
    worker_drifts = []

    for a_i, w_i in zip(alphas, w_arrs):
        diff = w_i - w_bar
        d_sq = float(np.dot(diff, diff))
        d_l2 = float(np.sqrt(max(0.0, d_sq)))
        norm_d = d_l2 / denom
        worker_drifts.append(norm_d)

        v_t += a_i * d_sq
        legacy_drift += a_i * norm_d
        if norm_d > m_t:
            m_t = norm_d

    v_t = max(0.0, v_t)
    q_t = v_t / denom_sq

    # 2. Pairwise consensus error identity:
    # V_pairwise = (1/2) * sum_i sum_j alpha_i * alpha_j * ||w_i - w_j||^2
    v_pairwise = 0.0
    for i in range(m):
        for j in range(m):
            diff_ij = w_arrs[i] - w_arrs[j]
            d_ij_sq = float(np.dot(diff_ij, diff_ij))
            v_pairwise += 0.5 * alphas[i] * alphas[j] * d_ij_sq

    discrepancy = abs(v_t - v_pairwise)

    metrics = ConsensusMetrics(
        v_t=v_t,
        q_t=q_t,
        m_t=m_t,
        norm_w_bar_sq=norm_w_bar_sq,
        norm_w_bar=norm_w_bar,
        local_drift_norm=worker_drifts[0],
        local_weight=float(alphas[0]),
        legacy_drift=legacy_drift,
    )

    return metrics, discrepancy

