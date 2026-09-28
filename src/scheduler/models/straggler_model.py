"""Straggler Penalty Model: P_straggler = lambda * std_runtime."""
import os
import json
from typing import Dict, Any, Optional


class StragglerPenaltyModel:
    """Calculates straggler penalty based on runtime variance and configurable lambda."""

    def __init__(
        self,
        lambda_penalty: float = 1.0,
        runtime_profile_path_or_dict: Optional[Any] = None,
    ):
        self.lambda_penalty = float(lambda_penalty)
        self.runtime_stats: Dict[str, Dict[str, float]] = {}

        if runtime_profile_path_or_dict is not None:
            if isinstance(runtime_profile_path_or_dict, str):
                if os.path.exists(runtime_profile_path_or_dict):
                    with open(runtime_profile_path_or_dict, "r", encoding="utf-8") as f:
                        self.runtime_stats = json.load(f)
            elif isinstance(runtime_profile_path_or_dict, dict):
                self.runtime_stats = runtime_profile_path_or_dict

    def update_node_stats(
        self,
        node_id: str,
        runtime_mean_ms: float,
        runtime_std_ms: float,
        runtime_ema_ms: Optional[float] = None,
    ) -> None:
        """Updates runtime statistics for a node."""
        self.runtime_stats[node_id] = {
            "runtime_mean_ms": float(runtime_mean_ms),
            "runtime_std_ms": float(runtime_std_ms),
            "runtime_ema_ms": float(runtime_ema_ms if runtime_ema_ms is not None else runtime_mean_ms),
        }

    def compute_penalty_ms(
        self,
        node_id: str,
        fallback_std_ms: float = 0.0,
    ) -> float:
        """Calculates P_straggler = lambda * std_runtime (in ms)."""
        node_stat = self.runtime_stats.get(node_id, {})
        std_ms = float(node_stat.get("runtime_std_ms", fallback_std_ms))
        penalty = self.lambda_penalty * max(0.0, std_ms)
        return float(penalty)

    def get_node_std_ms(self, node_id: str, fallback_std_ms: float = 0.0) -> float:
        """Returns empirical runtime std in ms for node."""
        return float(self.runtime_stats.get(node_id, {}).get("runtime_std_ms", fallback_std_ms))
