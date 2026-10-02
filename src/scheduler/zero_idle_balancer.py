"""
Zero-Idle / Iso-Time Workload Balancer for Heterogeneous Cluster.

Automatically determines the optimal per-node batch size allocation
for ANY arbitrary global batch size B (e.g. 256, 300, 368, 500, 640)
such that all workers finish compute at the same time:
    T_compute_0(b_0) ≈ T_compute_1(b_1) ≈ ... ≈ T_compute_4(b_4)
thereby minimizing GPU idle waiting time (GPU Idle -> 0ms).
"""

from typing import Dict, Optional, List, Tuple


# Empirical processing capacities in samples/second
DEFAULT_NODE_SPEEDS = {
    "lab01": 1100.0,  # 2x Titan Z GPU (MirroredStrategy)
    "lab05": 66.2,    # Intel Core i5/i7 (4 cores / high clock)
    "lab02": 27.5,    # Intel Pentium G3260 (2 cores)
    "lab03": 22.9,    # Intel Core i3-560 (2C/4T)
    "lab04": 22.9,    # Intel Core i3-560 (2C/4T)
}

# Physical Titan Z GPU VRAM bounds
GPU_MAX_BATCH = 448
GPU_MIN_BATCH = 64


class ZeroIdleBalancer:
    """Analytical Iso-Time Balancer for Heterogeneous Distributed Training."""

    @staticmethod
    def solve_optimal_allocation(
        global_batch_size: int,
        custom_speeds: Optional[Dict[str, float]] = None,
        slow_node_id: Optional[str] = None,
        slowdown_factor: float = 2.0,
        zero_out_straggler: bool = False,
    ) -> Dict[str, int]:
        """Calculates optimal batch sizes [b0, b1, b2, b3, b4] for arbitrary global batch size B.
        
        Args:
            global_batch_size: Target global mini-batch size (e.g. 300, 400, 500, 640).
            custom_speeds: Optional dictionary of node processing speeds in samples/sec.
            slow_node_id: Optional node ID experiencing runtime slowdown.
            slowdown_factor: Factor by which slow_node_id is degraded.
            zero_out_straggler: If True, drops the slow node completely (batch = 0).
            
        Returns:
            Dict mapping node IDs to integer batch sizes summing exactly to global_batch_size.
        """
        B = int(global_batch_size)
        if B % 2 != 0:
            B = B - 1
            print(f"[ZeroIdleBalancer] Warning: Odd global batch size {global_batch_size} adjusted to {B} for even tensor division.")
        if B < 32:
            raise ValueError(f"Global batch size {B} is too small for 5-node cluster (minimum 32).")

        speeds = dict(custom_speeds or DEFAULT_NODE_SPEEDS)
        node_order = ["lab01", "lab02", "lab03", "lab04", "lab05"]

        # Adjust for degraded straggler if specified
        if slow_node_id and slow_node_id in speeds:
            if zero_out_straggler:
                speeds[slow_node_id] = 0.0
            else:
                speeds[slow_node_id] = max(1.0, speeds[slow_node_id] / float(slowdown_factor))

        total_speed = sum(speeds.values())
        if total_speed <= 0.0:
            raise ValueError("Total compute speed must be positive.")

        # 1. Proportional floating point target batches
        raw_b = {n: (speeds[n] / total_speed) * B for n in node_order}

        # 2. Enforce GPU Titan Z VRAM physical limits
        if raw_b["lab01"] > GPU_MAX_BATCH:
            raw_b["lab01"] = float(GPU_MAX_BATCH)
            remaining_b = B - GPU_MAX_BATCH
            cpu_nodes = [n for n in node_order if n != "lab01" and speeds[n] > 0.0]
            tot_cpu_spd = sum(speeds[n] for n in cpu_nodes)
            if tot_cpu_spd > 0:
                for n in cpu_nodes:
                    raw_b[n] = (speeds[n] / tot_cpu_spd) * remaining_b

        # 3. Quantize to even integers (divisible by 2 for clean data/MirroredStrategy tensors)
        rounded = {}
        for n in node_order:
            if speeds[n] <= 0.0:
                rounded[n] = 0
            else:
                b_val = max(2, int(round(raw_b[n] / 2.0) * 2))
                rounded[n] = b_val

        # If straggler zero-out requested, ensure exactly 0
        if slow_node_id and zero_out_straggler and slow_node_id in rounded:
            rounded[slow_node_id] = 0

        # 4. Reconcile residual discrepancy to strictly match global_batch_size B
        diff = B - sum(rounded.values())
        if diff != 0:
            # Add/subtract diff to GPU if within bounds and diff is even
            if diff % 2 == 0 and (GPU_MIN_BATCH <= rounded["lab01"] + diff <= GPU_MAX_BATCH):
                rounded["lab01"] += diff
            else:
                # Distribute in steps of 2 across fastest healthy nodes
                step = 2 if diff > 0 else -2
                donor_order = [n for n in ["lab01", "lab05", "lab02", "lab04", "lab03"] if not (zero_out_straggler and n == slow_node_id)]
                idx = 0
                while sum(rounded.values()) != B and idx < 200:
                    target_n = donor_order[idx % len(donor_order)]
                    if rounded[target_n] + step >= (0 if target_n != "lab01" else GPU_MIN_BATCH):
                        if target_n == "lab01" and rounded[target_n] + step > GPU_MAX_BATCH:
                            pass
                        else:
                            rounded[target_n] += step
                    idx += 1

        # Final invariant verification
        assert sum(rounded.values()) == B, f"Internal solver error: allocation sum {sum(rounded.values())} != {B}"
        return {n: int(rounded[n]) for n in node_order}

    @staticmethod
    def estimate_per_node_timings(allocation: Dict[str, int], custom_speeds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Estimates per-node compute times in milliseconds."""
        speeds = custom_speeds or DEFAULT_NODE_SPEEDS
        timings = {}
        for n, b in allocation.items():
            s = speeds.get(n, 1.0)
            timings[n] = float((b / s) * 1000.0) if (b > 0 and s > 0) else 0.0
        return timings

