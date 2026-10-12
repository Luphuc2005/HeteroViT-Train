#!/usr/bin/env python3
"""
Minimal MPI Allreduce Probe for HeteroViT-MPI.
Measures runtime and verifies numerical correctness of Allreduce
on a 2,693,578 float32 array (~10.77 MB) across 5 nodes.
"""

import os
import sys
import time
import socket
import numpy as np
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    hostname = socket.gethostname()

    # Model parameter count for ViT-Tiny in HeteroViT-MPI
    N = 2693578
    dtype = np.float32

    # Allocate send and receive buffers
    send_val = float(rank + 1)
    send_buf = np.full(N, send_val, dtype=dtype)
    recv_buf = np.zeros(N, dtype=dtype)
    payload_bytes = send_buf.nbytes

    # Gather host info
    node_info = comm.gather((rank, hostname, send_val), root=0)

    if rank == 0:
        print("=" * 80)
        print(" HETEROVIT-MPI: ALLREDUCE RUNTIME ALGORITHM PROBE (5 NODES)")
        print("=" * 80)
        print(f" Communicator Size : {size} ranks")
        for r, h, v in node_info:
            print(f"   * Rank {r} on {h:10s} | Send Value: {v:.1f}")
        print(f" Buffer Elements   : {N:,} elements")
        print(f" Buffer Dtype      : {dtype.__name__}")
        print(f" Payload Size      : {payload_bytes:,} bytes ({payload_bytes / (1024*1024):.2f} MB)")
        print("-" * 80)

    # Warmup round
    comm.Barrier()
    comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
    comm.Barrier()

    # Timed rounds
    NUM_ROUNDS = 5
    times = []
    for rnd in range(NUM_ROUNDS):
        comm.Barrier()
        t0 = time.perf_counter()
        comm.Allreduce(send_buf, recv_buf, op=MPI.SUM)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    # Verification
    expected_sum = sum(range(1, size + 1))  # 1+2+3+4+5 = 15.0
    is_correct = bool(np.allclose(recv_buf, expected_sum, rtol=1e-5, atol=1e-5))

    avg_time = float(np.mean(times))
    min_time = float(np.min(times))
    max_time = float(np.max(times))

    # Gather timing metrics to rank 0
    all_timings = comm.gather((rank, hostname, avg_time, min_time, max_time, is_correct), root=0)

    if rank == 0:
        print(f" Results Verification: {'PASSED (Sum = ' + str(expected_sum) + ')' if is_correct else 'FAILED'}")
        print(" Measured Allreduce Times across 5 Nodes (over 5 rounds):")
        for r, h, avg_t, min_t, max_t, ok in all_timings:
            print(f"   * Rank {r} [{h:10s}]: Avg = {avg_t:6.2f} ms | Min = {min_t:6.2f} ms | Max = {max_t:6.2f} ms | Correct: {ok}")
        cluster_avg = float(np.mean([t[2] for t in all_timings]))
        throughput_mbps = (payload_bytes * 8 / (cluster_avg / 1000.0)) / 1e6
        print("-" * 80)
        print(f" Cluster Mean Allreduce Latency : {cluster_avg:.2f} ms")
        print(f" Effective Link Throughput       : {throughput_mbps:.2f} Mbps (on 100 Mbps network)")
        print("=" * 80)

if __name__ == "__main__":
    main()

