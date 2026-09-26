#!/usr/bin/env python3
"""Aggregates multi-seed results for V5 MAXPERF and prints statistical summary & ablation table."""
import glob
import os
import csv
import math

def calculate_mean_std(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(variance)
    return mean, std

def parse_run(run_dir):
    train_csv = os.path.join(run_dir, "train.csv")
    if not os.path.exists(train_csv):
        return None
    with open(train_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    
    epoch_times = [float(r["epoch_time"]) for r in rows]
    throughputs = [float(r["samples_per_sec"]) for r in rows]
    val_accs = [float(r["val_accuracy"]) * 100 for r in rows]
    gpu_steps = [float(r["avg_gpu_step_ms"]) for r in rows[1:]]
    cpu_steps = [float(r["avg_cpu_step_ms"]) for r in rows[1:]]
    gpu_idles = [float(r["avg_gpu_idle_ms"]) for r in rows[1:]]
    
    total_time_min = sum(epoch_times) / 60.0
    ss_epoch_time = sum(epoch_times[1:]) / len(epoch_times[1:])
    ss_throughput = sum(throughputs[1:]) / len(throughputs[1:])
    ss_gpu_step = sum(gpu_steps) / len(gpu_steps)
    ss_cpu_step = sum(cpu_steps) / len(cpu_steps)
    ss_gpu_idle = sum(gpu_idles) / len(gpu_idles)
    best_val_acc = max(val_accs)
    
    return {
        "run_dir": os.path.basename(run_dir),
        "total_time_min": total_time_min,
        "epoch_time": ss_epoch_time,
        "throughput": ss_throughput,
        "gpu_step": ss_gpu_step,
        "cpu_step": ss_cpu_step,
        "gpu_idle": ss_gpu_idle,
        "best_val_acc": best_val_acc,
        "epochs": len(rows),
    }

def main():
    base_dir = "./results/joint_benchmarks/isobatch256_gpu232_cpu24_maxperf"
    run_dirs = sorted(glob.glob(os.path.join(base_dir, "joint_*")))
    
    runs = []
    for rd in run_dirs:
        data = parse_run(rd)
        if data:
            runs.append(data)
            
    print(f"\n====================================================================================")
    print(f"       V5 MAXPERF MULTI-SEED STATISTICAL SUMMARY ({len(runs)} Run(s) Found)")
    print(f"====================================================================================")
    
    for i, r in enumerate(runs, 1):
        print(f"  Run {i} [{r['run_dir'][:42]}]:")
        print(f"    • Steady Epoch Time : {r['epoch_time']:5.2f} s | Throughput: {r['throughput']:6.1f} img/s")
        print(f"    • GPU Step: {r['gpu_step']:5.1f} ms | CPU Step: {r['cpu_step']:5.1f} ms | GPU Idle: {r['gpu_idle']:4.1f} ms")
        print(f"    • Best Val Acc: {r['best_val_acc']:5.2f}% | Total Time: {r['total_time_min']:5.2f} min ({r['epochs']} epochs)\n")

    if not runs:
        print("No runs found in directory:", base_dir)
        return

    # Aggregate stats
    m_ep, s_ep = calculate_mean_std([r["epoch_time"] for r in runs])
    m_thr, s_thr = calculate_mean_std([r["throughput"] for r in runs])
    m_gpu, s_gpu = calculate_mean_std([r["gpu_step"] for r in runs])
    m_cpu, s_cpu = calculate_mean_std([r["cpu_step"] for r in runs])
    m_idle, s_idle = calculate_mean_std([r["gpu_idle"] for r in runs])
    m_acc, s_acc = calculate_mean_std([r["best_val_acc"] for r in runs])
    m_tot, s_tot = calculate_mean_std([r["total_time_min"] for r in runs])

    print("------------------------------------------------------------------------------------")
    print(f"  >>> AGGREGATED BENCHMARK (Mean ± Std, N={len(runs)}):")
    print(f"      • Throughput      : {m_thr:6.1f} ± {s_thr:4.1f} img/s")
    print(f"      • Epoch Time      : {m_ep:5.2f} ± {s_ep:4.2f} s")
    print(f"      • Total 20 Epochs : {m_tot:5.2f} ± {s_tot:4.2f} min")
    print(f"      • GPU Step Time   : {m_gpu:5.1f} ± {s_gpu:4.1f} ms")
    print(f"      • CPU Step Time   : {m_cpu:5.1f} ± {s_cpu:4.1f} ms")
    print(f"      • GPU Idle Time   : {m_idle:5.1f} ± {s_idle:4.1f} ms")
    print(f"      • Best Val Acc    : {m_acc:5.2f} ± {s_acc:4.2f} %")
    print("------------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
