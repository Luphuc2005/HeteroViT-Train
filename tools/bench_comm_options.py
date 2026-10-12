import time
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()

N = 2_693_578 # ViT-Tiny parameter count

# Warmup
warm_buf = np.zeros(1000, dtype=np.float32)
comm.Allreduce(MPI.IN_PLACE, warm_buf, op=MPI.SUM)

# 1. FP32 Allreduce (Baseline)
fp32_w = np.random.randn(N).astype(np.float32)
comm.Barrier()
t0 = time.perf_counter()
fp32_recv = np.empty_like(fp32_w)
comm.Allreduce(fp32_w, fp32_recv, op=MPI.SUM)
comm.Barrier()
t_fp32 = (time.perf_counter() - t0) * 1000.0

# 2. FP16 Allgather + Local FP32 Accumulation
comm.Barrier()
t_down_0 = time.perf_counter()
fp16_w = fp32_w.astype(np.float16)
t_down = (time.perf_counter() - t_down_0) * 1000.0

send_bytes = np.ascontiguousarray(fp16_w).view(np.uint8)
recv_bytes = np.empty(world_size * send_bytes.nbytes, dtype=np.uint8)

comm.Barrier()
t_comm_0 = time.perf_counter()
comm.Allgather(send_bytes, recv_bytes)
comm.Barrier()
t_ag_comm = (time.perf_counter() - t_comm_0) * 1000.0

t_up_0 = time.perf_counter()
chunk_nbytes = send_bytes.nbytes
acc_fp32 = np.zeros(N, dtype=np.float32)
sample_weights = [10640.0, 1340.0, 1340.0, 1340.0, 1340.0]
total_samples = sum(sample_weights)
for r in range(world_size):
    offset = r * chunk_nbytes
    chunk_fp16 = np.frombuffer(recv_bytes[offset:offset + chunk_nbytes], dtype=np.float16)
    weight_ratio = np.float32(sample_weights[r] / total_samples)
    acc_fp32 += chunk_fp16.astype(np.float32) * weight_ratio
t_up = (time.perf_counter() - t_up_0) * 1000.0
total_fp16_ag = t_down + t_ag_comm + t_up

# 3. FP16 Allreduce via MPI.Op callback (normalized weights)
def fp16_sum_op(invec, inoutvec, datatype):
    a = np.frombuffer(invec, dtype=np.float16)
    b = np.frombuffer(inoutvec, dtype=np.float16)
    b[:] += a

custom_op = MPI.Op.Create(fp16_sum_op, commute=True)
comm.Barrier()
t_op_down_0 = time.perf_counter()
# Weight before sending
weight_ratio = np.float16(sample_weights[rank] / total_samples)
fp16_weighted = (fp32_w.astype(np.float32) * float(weight_ratio)).astype(np.float16)
t_op_down = (time.perf_counter() - t_op_down_0) * 1000.0

fp16_red_recv = np.empty(N, dtype=np.float16)
comm.Barrier()
t_op_comm_0 = time.perf_counter()
comm.Allreduce([fp16_weighted, MPI.SHORT], [fp16_red_recv, MPI.SHORT], op=custom_op)
comm.Barrier()
t_op_comm = (time.perf_counter() - t_op_comm_0) * 1000.0

t_op_up_0 = time.perf_counter()
fp32_res_op = fp16_red_recv.astype(np.float32)
t_op_up = (time.perf_counter() - t_op_up_0) * 1000.0
total_fp16_op = t_op_down + t_op_comm + t_op_up
custom_op.Free()

if rank == 0:
    print(f"\n================================================================================")
    print(f" COMM PRECISION BENCHMARK (5 NODES, 100 Mbps ETHERNET)")
    print(f"================================================================================")
    print(f" Elements:                        {N:,}")
    print(f" FP32 payload:                    {fp32_w.nbytes / (1024*1024):.2f} MB")
    print(f" FP16 payload:                    {fp16_w.nbytes / (1024*1024):.2f} MB (-50.0%)")
    print(f"--------------------------------------------------------------------------------")
    print(f" [1] Baseline FP32 Allreduce:     {t_fp32:6.1f} ms")
    print(f"--------------------------------------------------------------------------------")
    print(f" [2] FP16 Allgather + FP32 Accum: {total_fp16_ag:6.1f} ms  (Comm: {t_ag_comm:5.1f}ms, Down: {t_down:4.1f}ms, Up: {t_up:4.1f}ms)")
    print(f" [3] FP16 Allreduce + Custom Op:  {total_fp16_op:6.1f} ms  (Comm: {t_op_comm:5.1f}ms, Down: {t_op_down:4.1f}ms, Up: {t_op_up:4.1f}ms)")
    print(f"================================================================================\n")
