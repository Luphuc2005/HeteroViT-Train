"""Profiles MPI AllReduce collective communication time across communicator subsets."""
import os
import sys
import json
import time
import argparse
import numpy as np
from mpi4py import MPI

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_JSON = os.path.join(PROJECT_ROOT, "profiles", "subset_allreduce_benchmarks.json")

RANK_TO_NODE = {0: "lab01", 1: "lab02", 2: "lab03", 3: "lab04", 4: "lab05"}


def benchmark_communicator(comm: MPI.Comm, payload_size_mb: float = 10.275, num_warmup: int = 5, num_trials: int = 20):
    """Measures MPI AllReduce latency and bandwidth on given sub-communicator."""
    if comm == MPI.COMM_NULL:
        return None

    size_bytes = int(payload_size_mb * 1024 * 1024)
    num_floats = size_bytes // 4
    sendbuf = np.ones(num_floats, dtype=np.float32)
    recvbuf = np.empty_like(sendbuf)

    # Warmup
    for _ in range(num_warmup):
        comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)

    times_ms = []
    for _ in range(num_trials):
        comm.Barrier()
        t0 = time.perf_counter()
        comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times_ms))
    std_ms = float(np.std(times_ms))
    return {
        "mean_time_ms": mean_ms,
        "std_time_ms": std_ms,
        "trials": times_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile AllReduce on Communicator Subsets")
    parser.add_argument("--payload-mb", type=float, default=10.2751846, help="Gradient payload size in MB")
    parser.add_argument("--trials", type=int, default=20, help="Number of benchmark repetitions")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_JSON, help="Output JSON path")
    args = parser.parse_args()

    world_comm = MPI.COMM_WORLD
    world_rank = world_comm.Get_rank()
    world_size = world_comm.Get_size()

    subsets = {
        "5nodes_lab01_lab02_lab03_lab04_lab05": {
            "label": "5-node Full Cluster",
            "ranks": [0, 1, 2, 3, 4],
        },
        "4nodes_lab01_lab03_lab04_lab05": {
            "label": "4-node (drop lab02 / rank 1)",
            "ranks": [0, 2, 3, 4],
        },
        "4nodes_lab01_lab02_lab04_lab05": {
            "label": "4-node (drop lab03 / rank 2)",
            "ranks": [0, 1, 3, 4],
        },
        "4nodes_lab01_lab02_lab03_lab05": {
            "label": "4-node (drop lab04 / rank 3)",
            "ranks": [0, 1, 2, 4],
        },
        "4nodes_lab01_lab02_lab03_lab04": {
            "label": "4-node (drop lab05 / rank 4)",
            "ranks": [0, 1, 2, 3],
        },
        "3nodes_lab01_lab02_lab05": {
            "label": "3-node (lab01, lab02, lab05)",
            "ranks": [0, 1, 4],
        },
        "3nodes_lab01_lab04_lab05": {
            "label": "3-node (lab01, lab04, lab05)",
            "ranks": [0, 3, 4],
        },
    }

    results = {}
    for subset_key, s_info in subsets.items():
        ranks_in_subset = s_info["ranks"]
        color = 1 if world_rank in ranks_in_subset else MPI.UNDEFINED
        sub_comm = world_comm.Split(color, world_rank)

        bench = None
        if color == 1:
            bench = benchmark_communicator(sub_comm, payload_size_mb=args.payload_mb, num_trials=args.trials)
            sub_comm.Free()

        gathered = world_comm.gather(bench, root=0)

        if world_rank == 0 and gathered:
            valid_benches = [b for b in gathered if b is not None]
            if valid_benches:
                first = valid_benches[0]
                results[subset_key] = {
                    "label": s_info["label"],
                    "ranks": ranks_in_subset,
                    "nodes": [RANK_TO_NODE[r] for r in ranks_in_subset],
                    "size": len(ranks_in_subset),
                    "payload_mb": args.payload_mb,
                    "mean_time_ms": first["mean_time_ms"],
                    "std_time_ms": first["std_time_ms"],
                }
                print(f"[{s_info['label']}] mean: {first['mean_time_ms']:.2f} ms | std: {first['std_time_ms']:.2f} ms")

    if world_rank == 0:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Benchmark results successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
