"""Compute Cost Model based on empirical node profiles."""
import os
import json
import math
from typing import Dict, Any, Optional, Mapping


class ComputeCostModel:
    """Predicts T_compute(node, batch) using empirical measurements and interpolation."""

    def __init__(self, profile_path_or_dict: Any):
        if isinstance(profile_path_or_dict, str):
            if not os.path.exists(profile_path_or_dict):
                raise FileNotFoundError(f"Compute profile not found at {profile_path_or_dict}")
            with open(profile_path_or_dict, "r", encoding="utf-8") as f:
                self.profile = json.load(f)
        elif isinstance(profile_path_or_dict, dict):
            self.profile = profile_path_or_dict
        else:
            raise TypeError("profile must be a file path string or dictionary")

        self.node_batches: Dict[str, Dict[int, float]] = {}
        self.node_std: Dict[str, Dict[int, float]] = {}
        self.node_throughput: Dict[str, Dict[int, float]] = {}

        for node_id, data in self.profile.items():
            b_dict = {}
            s_dict = {}
            t_dict = {}
            for b_str, p in data.get("profiles", {}).items():
                b = int(b_str)
                b_dict[b] = float(p.get("mean_compute_ms", 0.0))
                s_dict[b] = float(p.get("std_compute_ms", 0.0))
                t_dict[b] = float(p.get("throughput_img_s", 0.0))
            self.node_batches[node_id] = b_dict
            self.node_std[node_id] = s_dict
            self.node_throughput[node_id] = t_dict

    def predict_compute_ms(self, node_id: str, batch_size: int) -> float:
        """Predicts compute time in milliseconds for node_id at batch_size.
        
        If batch_size == 0, returns 0.0 ms (inactive node).
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            return 0.0

        if node_id not in self.node_batches:
            raise KeyError(f"Node '{node_id}' not found in compute profile.")

        b_map = self.node_batches[node_id]
        if not b_map:
            raise ValueError(f"No profile entries for node '{node_id}'.")

        # 1. Exact match
        if batch_size in b_map:
            return b_map[batch_size]

        # 2. Linear interpolation between bounding sizes
        sorted_sizes = sorted(b_map.keys())
        lower = [s for s in sorted_sizes if s < batch_size]
        upper = [s for s in sorted_sizes if s > batch_size]

        if lower and upper:
            b_lo, b_hi = lower[-1], upper[0]
            t_lo, t_hi = b_map[b_lo], b_map[b_hi]
            ratio = (batch_size - b_lo) / float(b_hi - b_lo)
            return t_lo + ratio * (t_hi - t_lo)

        # 3. Extrapolation via throughput of nearest reference
        ref_b = lower[-1] if lower else upper[0]
        ref_t = b_map[ref_b]
        tput = float(ref_b) / (ref_t / 1000.0)
        return (float(batch_size) / max(tput, 1e-6)) * 1000.0

    def get_compute_std_ms(self, node_id: str, batch_size: int) -> float:
        """Gets or estimates compute standard deviation in milliseconds."""
        if batch_size <= 0 or node_id not in self.node_std:
            return 0.0
        s_map = self.node_std[node_id]
        if batch_size in s_map:
            return s_map[batch_size]
        sorted_sizes = sorted(s_map.keys())
        closest = min(sorted_sizes, key=lambda s: abs(s - batch_size))
        return s_map.get(closest, 0.0)
