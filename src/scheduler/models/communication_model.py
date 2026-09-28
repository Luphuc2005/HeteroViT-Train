"""Communication Cost Model based on network measurements and MPI AllReduce."""
import os
import json
from typing import Dict, Any, List, Optional, Set


class CommunicationCostModel:
    """Predicts T_comm for active nodes in distributed gradient AllReduce."""

    def __init__(self, profile_path_or_dict: Any, default_model_size_mb: float = 10.28):
        self.default_model_size_mb = float(default_model_size_mb)
        if isinstance(profile_path_or_dict, str):
            if os.path.exists(profile_path_or_dict):
                with open(profile_path_or_dict, "r", encoding="utf-8") as f:
                    self.profile = json.load(f)
            else:
                self.profile = {}
        elif isinstance(profile_path_or_dict, dict):
            self.profile = profile_path_or_dict
        else:
            self.profile = {}

        self.model_size_mb = float(self.profile.get("model_size_mb", self.default_model_size_mb))
        self.allreduce_data = self.profile.get("allreduce", {})
        self.p2p_data = self.profile.get("p2p_pairs", {})

        # Compute average pair latency and effective bandwidth
        latencies = []
        bws = []
        for pair_name, p_info in self.p2p_data.items():
            meas = p_info.get("measurements", {})
            if "64B" in meas:
                latencies.append(meas["64B"]["latency_ms"])
            if "10MB" in meas:
                bws.append(meas["10MB"]["bw_mb_s"])
            elif "5MB" in meas:
                bws.append(meas["5MB"]["bw_mb_s"])

        self.avg_latency_ms = float(sum(latencies) / len(latencies)) if latencies else 0.13
        self.avg_bw_mb_s = float(sum(bws) / len(bws)) if bws else 11.2  # ~100 Mbps

    def predict_comm_ms(
        self,
        active_nodes: List[str],
        data_size_mb: Optional[float] = None,
    ) -> float:
        """Predicts MPI AllReduce communication time in ms for given active nodes.
        
        Args:
            active_nodes: List or set of active node names (b_i > 0).
            data_size_mb: Model/gradient payload size in MB.
        """
        n_active = len(active_nodes)
        if n_active <= 1:
            return 0.0

        size_mb = float(data_size_mb or self.model_size_mb)

        # 1. Direct empirical AllReduce profile match if available
        if n_active == 5 and "5_nodes" in self.allreduce_data:
            return float(self.allreduce_data["5_nodes"]["mean_time_ms"])

        if n_active == 4:
            if "lab02" not in active_nodes and "4_nodes_no_lab02" in self.allreduce_data:
                return float(self.allreduce_data["4_nodes_no_lab02"]["mean_time_ms"])
            if "lab03" not in active_nodes and "4_nodes_no_lab03" in self.allreduce_data:
                return float(self.allreduce_data["4_nodes_no_lab03"]["mean_time_ms"])

        # 2. Ring / Tree AllReduce communication model:
        # T_comm = latency + 2 * ((N - 1) / N) * (data_size / BW_eff)
        ring_factor = 2.0 * (n_active - 1) / float(n_active)
        transfer_time_s = (ring_factor * size_mb) / max(self.avg_bw_mb_s, 1e-6)
        t_comm_ms = self.avg_latency_ms + (transfer_time_s * 1000.0)

        return float(t_comm_ms)
