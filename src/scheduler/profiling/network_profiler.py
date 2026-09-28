"""MPI-based Network Profiler for 5-Node Heterogeneous Cluster.

Measures:
1. Point-to-point ping-pong latency (64B) and effective bandwidth across all pairs
   for multiple payloads: 64KB, 1MB, 5MB, 10MB, 20MB, 50MB.
2. Actual MPI AllReduce collective communication time for the 10.28MB ViT model
   across 5 nodes and 4 nodes (sub-communicator without stragglers).
3. Saves empirical results to profiles/network_profile.json.
"""
import os
import sys
import time
import json
import socket
import argparse
from typing import Dict, Any, List
import numpy as np
from mpi4py import MPI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


NODE_NAMES = {
    0: "lab01",
    1: "lab02",
    2: "lab03",
    3: "lab04",
    4: "lab05",
}

# Payloads for point-to-point profiling
PAYLOAD_SIZES = {
    "64B": 64,
    "64KB": 64 * 1024,
    "1MB": 1024 * 1024,
    "5MB": 5 * 1024 * 1024,
    "10MB": 10 * 1024 * 1024,
    "20MB": 20 * 1024 * 1024,
    "50MB": 50 * 1024 * 1024,
}

# Real model ViT-Tiny gradient buffer size (float32 elements)
VIT_PARAM_COUNT = 2693578  # 10,774,312 bytes (~10.28 MB)


def profile_p2p_pair(comm: MPI.Comm, src_rank: int, dst_rank: int, my_rank: int) -> Dict[str, Any]:
    """Measures latency and effective bandwidth between src_rank and dst_rank."""
    pair_results = {}

    for name, num_bytes in PAYLOAD_SIZES.items():
        count = max(1, num_bytes // 4)
        send_buf = np.ones(count, dtype=np.float32)
        recv_buf = np.empty(count, dtype=np.float32)

        if num_bytes <= 64:
            warmup_iters = 3
            bench_iters = 10
        elif num_bytes <= 1024 * 1024:
            warmup_iters = 2
            bench_iters = 4
        elif num_bytes <= 10 * 1024 * 1024:
            warmup_iters = 1
            bench_iters = 3
        else:
            warmup_iters = 1
            bench_iters = 2

        # Warmup
        for _ in range(warmup_iters):
            if my_rank == src_rank:
                comm.Send(send_buf, dest=dst_rank, tag=10)
                comm.Recv(recv_buf, source=dst_rank, tag=20)
            elif my_rank == dst_rank:
                comm.Recv(recv_buf, source=src_rank, tag=10)
                comm.Send(send_buf, dest=src_rank, tag=20)

        comm.Barrier()

        # Measurement
        durations = []
        for _ in range(bench_iters):
            if my_rank == src_rank:
                t0 = time.perf_counter()
                comm.Send(send_buf, dest=dst_rank, tag=10)
                comm.Recv(recv_buf, source=dst_rank, tag=20)
                t1 = time.perf_counter()
                durations.append(t1 - t0)
            elif my_rank == dst_rank:
                comm.Recv(recv_buf, source=src_rank, tag=10)
                comm.Send(send_buf, dest=src_rank, tag=20)

        comm.Barrier()

        if my_rank == src_rank:
            rtt_mean_s = float(np.mean(durations))
            rtt_std_s = float(np.std(durations))
            one_way_latency_ms = (rtt_mean_s / 2.0) * 1000.0
            total_bytes_transferred = 2 * num_bytes
            bw_mb_s = (total_bytes_transferred / (1024 * 1024)) / max(rtt_mean_s, 1e-9)

            pair_results[name] = {
                "bytes": num_bytes,
                "rtt_ms": rtt_mean_s * 1000.0,
                "latency_ms": one_way_latency_ms,
                "bw_mb_s": bw_mb_s,
                "std_rtt_ms": rtt_std_s * 1000.0,
            }
            if my_rank == 0 or dst_rank == 0:
                print(f"    payload={name:<5} -> latency={one_way_latency_ms:6.2f} ms | bw={bw_mb_s:6.2f} MB/s", flush=True)

    return pair_results


def profile_allreduce_communicator(comm_sub: MPI.Comm, my_sub_rank: int, buf_size_elements: int, label: str) -> Dict[str, Any]:
    """Measures collective AllReduce latency and bandwidth for a communicator."""
    send_buf = np.ones(buf_size_elements, dtype=np.float32)
    recv_buf = np.empty(buf_size_elements, dtype=np.float32)

    sub_size = comm_sub.Get_size()
    total_bytes = buf_size_elements * 4
    mb_size = total_bytes / (1024 * 1024)

    warmup_iters = 5
    bench_iters = 20

    for _ in range(warmup_iters):
        comm_sub.Allreduce(send_buf, recv_buf, op=MPI.SUM)

    comm_sub.Barrier()

    durations = []
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        comm_sub.Allreduce(send_buf, recv_buf, op=MPI.SUM)
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)

    comm_sub.Barrier()

    mean_ms = float(np.mean(durations))
    std_ms = float(np.std(durations))
    min_ms = float(np.min(durations))
    max_ms = float(np.max(durations))

    bus_factor = 2.0 * (sub_size - 1) / max(sub_size, 1)
    effective_bw_mb_s = (bus_factor * mb_size) / max(mean_ms / 1000.0, 1e-9)

    return {
        "label": label,
        "world_size": sub_size,
        "payload_bytes": total_bytes,
        "payload_mb": mb_size,
        "mean_time_ms": mean_ms,
        "std_time_ms": std_ms,
        "min_time_ms": min_ms,
        "max_time_ms": max_ms,
        "effective_bw_mb_s": effective_bw_mb_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Distributed Network Profiler for 5 Nodes")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "profiles", "network_profile.json"),
        help="Path to save output JSON profile",
    )
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    if rank == 0:
        print("=" * 80)
        print(" HETEROVIT-MPI: DISTRIBUTED NETWORK PROFILER (PHASE 1)")
        print(f" Participating Ranks: {world_size} nodes")
        print("=" * 80)

    all_pairs_results = {}
    for i in range(world_size):
        for j in range(i + 1, world_size):
            node_i = NODE_NAMES.get(i, f"rank_{i}")
            node_j = NODE_NAMES.get(j, f"rank_{j}")
            pair_key = f"{node_i}_{node_j}"

            if rank == 0:
                print(f"[*] Profiling pair: {node_i} (rank {i}) <-> {node_j} (rank {j})...", flush=True)

            pair_res = profile_p2p_pair(comm, i, j, rank)
            pair_res = comm.bcast(pair_res, root=i)
            if rank == 0:
                all_pairs_results[pair_key] = {
                    "node_a": node_i,
                    "node_b": node_j,
                    "rank_a": i,
                    "rank_b": j,
                    "measurements": pair_res,
                }
            comm.Barrier()

    allreduce_results = {}
    if rank == 0:
        print("-" * 80)
        print(f"[*] Profiling Collective MPI AllReduce on ViT-Tiny ({VIT_PARAM_COUNT * 4 / (1024*1024):.2f} MB)...")

    # 5-node AllReduce
    res_5node = profile_allreduce_communicator(comm, rank, VIT_PARAM_COUNT, "5-node (lab01..lab05)")
    if rank == 0:
        allreduce_results["5_nodes"] = res_5node
        print(f"  -> 5-node AllReduce time: {res_5node['mean_time_ms']:.2f} ms +/- {res_5node['std_time_ms']:.2f} ms (eff_bw: {res_5node['effective_bw_mb_s']:.1f} MB/s)")

    # 4-node AllReduce excluding rank 1 (lab02)
    color_no_lab02 = 1 if rank != 1 else MPI.UNDEFINED
    comm_no_lab02 = comm.Split(color=color_no_lab02, key=rank)
    if color_no_lab02 != MPI.UNDEFINED:
        res_4node_no_lab02 = profile_allreduce_communicator(
            comm_no_lab02, comm_no_lab02.Get_rank(), VIT_PARAM_COUNT, "4-node (excluding lab02)"
        )
        if rank == 0:
            allreduce_results["4_nodes_no_lab02"] = res_4node_no_lab02
            print(f"  -> 4-node (no lab02) AllReduce: {res_4node_no_lab02['mean_time_ms']:.2f} ms +/- {res_4node_no_lab02['std_time_ms']:.2f} ms (eff_bw: {res_4node_no_lab02['effective_bw_mb_s']:.1f} MB/s)")
        comm_no_lab02.Free()
    comm.Barrier()

    # 4-node AllReduce excluding rank 2 (lab03)
    color_no_lab03 = 1 if rank != 2 else MPI.UNDEFINED
    comm_no_lab03 = comm.Split(color=color_no_lab03, key=rank)
    if color_no_lab03 != MPI.UNDEFINED:
        res_4node_no_lab03 = profile_allreduce_communicator(
            comm_no_lab03, comm_no_lab03.Get_rank(), VIT_PARAM_COUNT, "4-node (excluding lab03)"
        )
        if rank == 0:
            allreduce_results["4_nodes_no_lab03"] = res_4node_no_lab03
            print(f"  -> 4-node (no lab03) AllReduce: {res_4node_no_lab03['mean_time_ms']:.2f} ms +/- {res_4node_no_lab03['std_time_ms']:.2f} ms (eff_bw: {res_4node_no_lab03['effective_bw_mb_s']:.1f} MB/s)")
        comm_no_lab03.Free()
    comm.Barrier()

    if rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        final_profile = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "vit_tiny",
            "model_size_mb": VIT_PARAM_COUNT * 4 / (1024 * 1024),
            "p2p_pairs": all_pairs_results,
            "allreduce": allreduce_results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_profile, f, indent=2)
        print("=" * 80)
        print(f"[SUCCESS] Network profile saved to: {args.output}")
        print("=" * 80)


if __name__ == "__main__":
    main()
