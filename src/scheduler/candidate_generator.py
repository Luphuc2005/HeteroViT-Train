"""Candidate Generator for 5-Node Heterogeneous Training Allocations."""
from typing import Dict, List, Optional, Set, Tuple


# Available batch options per node bounded by empirical profiling and physical hardware
NODE_BATCH_OPTIONS = {
    "lab01": [256, 320, 384, 416, 448], # Titan Z GPU memory limits (max 448 with grad accum)
    "lab02": [0, 8, 16, 24, 32, 48, 64],
    "lab03": [0, 8, 16, 24, 32, 48, 64],
    "lab04": [0, 8, 16, 24, 32, 48, 64],
    "lab05": [0, 16, 32, 48, 64, 96, 128], # Core i7 multi-core CPU
}


class CandidateGenerator:
    """Generates candidate workload allocations for heterogeneous 5-node cluster."""

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
            # C2: Heterogeneous Static 5-node (640 total)
            {
                "lab01": 384,
                "lab02": 64,
                "lab03": 48,
                "lab04": 48,
                "lab05": 96,
            },
            # C3: Balanced 5-node (Workload proportional to empirical compute capacities: 368 total)
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

    @staticmethod
    def generate_candidates(
        global_batch_size: int,
        current_allocation: Optional[Dict[str, int]] = None,
        custom_node_options: Optional[Dict[str, List[int]]] = None,
        slow_node_id: Optional[str] = None,
        nominal_allocation: Optional[Dict[str, int]] = None,
    ) -> List[Dict[str, int]]:
        """Generates valid candidate allocations that sum to global_batch_size.

        Supports dynamic workload shifting from a slow node to GPU/fast CPU,
        as well as evaluating dropping a straggler node (b_i = 0).
        """
        options = custom_node_options or NODE_BATCH_OPTIONS
        node_order = ["lab01", "lab02", "lab03", "lab04", "lab05"]

        candidates_map: Dict[Tuple[int, ...], Dict[str, int]] = {}

        min_gpu_batch = min(64, int(global_batch_size * 0.4))

        def add_candidate(alloc: Dict[str, int]):
            if sum(alloc.values()) != global_batch_size:
                return
            # lab01 (GPU) must stay active with minimum viable batch
            if alloc.get("lab01", 0) < min_gpu_batch:
                return
            # All batches must be non-negative
            if any(v < 0 for v in alloc.values()):
                return
            key = tuple(alloc.get(n, 0) for n in node_order)
            if key not in candidates_map:
                candidates_map[key] = {n: alloc.get(n, 0) for n in node_order}

        # 1. Always include current allocation
        if current_allocation is not None:
            add_candidate(current_allocation)

        # 2. Add Iso-Time Zero-Idle Candidates for arbitrary global batch size
        try:
            from src.scheduler.zero_idle_balancer import ZeroIdleBalancer
            c_iso = ZeroIdleBalancer.solve_optimal_allocation(global_batch_size)
            add_candidate(c_iso)
            if slow_node_id:
                c_iso_deg = ZeroIdleBalancer.solve_optimal_allocation(
                    global_batch_size, slow_node_id=slow_node_id, slowdown_factor=2.0
                )
                add_candidate(c_iso_deg)
                c_iso_zero = ZeroIdleBalancer.solve_optimal_allocation(
                    global_batch_size, slow_node_id=slow_node_id, zero_out_straggler=True
                )
                add_candidate(c_iso_zero)
        except Exception:
            pass

        # 3. If current allocation provided, generate focused rebalancing candidates
        if current_allocation is not None:
            target_slow_nodes = [slow_node_id] if slow_node_id else [n for n in node_order if n != "lab01"]
            for s_node in target_slow_nodes:
                curr_b = current_allocation.get(s_node, 0)
                if curr_b <= 0:
                    continue

                # Possible reduced batch sizes for the slow node
                avail_opts = set(options.get(s_node, []))
                avail_opts.add(0)
                if curr_b > 2:
                    avail_opts.add(curr_b // 2)
                candidate_reduced_batches = sorted([b for b in avail_opts if b < curr_b])

                for new_b in candidate_reduced_batches:
                    delta = curr_b - new_b
                    # Option A: Shift all delta to lab01 (GPU)
                    c_to_gpu = dict(current_allocation)
                    c_to_gpu[s_node] = new_b
                    c_to_gpu["lab01"] += delta
                    add_candidate(c_to_gpu)

                    # Option B: Shift all delta to lab05 (Core i7)
                    if current_allocation["lab05"] + delta <= max(options.get("lab05", [96])):
                        c_to_lab5 = dict(current_allocation)
                        c_to_lab5[s_node] = new_b
                        c_to_lab5["lab05"] += delta
                        add_candidate(c_to_lab5)

                    # Option C: Split delta between lab01 and lab05
                    half1 = (delta // 2)
                    half2 = delta - half1
                    c_split = dict(current_allocation)
                    c_split[s_node] = new_b
                    c_split["lab01"] += half1
                    c_split["lab05"] += half2
                    add_candidate(c_split)

            # If a worker was previously reduced (b < nominal) and recovering, test restoring its batch
            nom = nominal_allocation or {}
            for node in ["lab02", "lab03", "lab04", "lab05"]:
                curr_b = current_allocation.get(node, 0)
                default_nom = 48 if node in ("lab03", "lab04") else (64 if node == "lab02" else 96)
                nominal_b = nom.get(node, default_nom)
                if curr_b < nominal_b:
                    delta = nominal_b - curr_b
                    # Take delta back from lab01
                    if current_allocation["lab01"] - delta >= min_gpu_batch:
                        c_restore = dict(current_allocation)
                        c_restore[node] = nominal_b
                        c_restore["lab01"] -= delta
                        add_candidate(c_restore)

        # 3. Add predefined Phase 1 candidates if matching global batch
        for b_cand in CandidateGenerator.get_benchmark_candidates():
            if sum(b_cand.values()) == global_batch_size:
                add_candidate(b_cand)

        return list(candidates_map.values())
