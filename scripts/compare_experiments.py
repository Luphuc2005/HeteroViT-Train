#!/usr/bin/env python3
"""
HeteroViT-MPI: Experiment Comparison & Benchmark Summary Tool
Parses results from results/ directory and produces a comparative Markdown table:
- Experiment Name & Mode (AllReduce vs Local SGD H)
- 30-Epoch Wall-Clock Duration (s / mins)
- Speedup vs Baseline
- Cluster Throughput (img/s)
- Total Communication Rounds & Bytes (MB)
- Best Validation Accuracy (%)
- Final Test Accuracy (%)
"""

import os
import glob
import json
import csv
import re
import argparse
from typing import Dict, List, Any, Optional


def parse_experiment_dir(exp_dir: str) -> Optional[Dict[str, Any]]:
    meta_file = os.path.join(exp_dir, "metadata.json")
    train_csv = os.path.join(exp_dir, "train.csv")
    run_log = os.path.join(exp_dir, "run.log")

    if not os.path.exists(run_log):
        return None

    name = os.path.basename(exp_dir)
    total_time_s = None
    best_val_acc = None
    final_test_acc = None
    comm_rounds = None
    avg_tput = None
    epochs_completed = 0
    sync_mode = "unknown"
    H_val = "1"
    policy = "allreduce"

    # 1. Parse run.log for key benchmarks
    with open(run_log, "r", encoding="utf-8", errors="ignore") as f:
        log_content = f.read()

        # Sync mode & H
        m_mode = re.search(r"Sync Mode\s*:\s*([^\n]+)", log_content)
        if m_mode:
            sync_mode = m_mode.group(1).strip()

        m_h = re.search(r"Synchronization H\s*:\s*(\d+)", log_content)
        if m_h:
            H_val = m_h.group(1).strip()

        m_pol = re.search(r"Averaging Policy\s*:\s*([^\n]+)", log_content)
        if m_pol:
            policy = m_pol.group(1).strip()

        # Comm rounds
        m_comm = re.search(r"Total Comm Rounds:\s*(\d+)", log_content)
        if m_comm:
            comm_rounds = int(m_comm.group(1))

        # Total training time
        m_time = re.search(r"Total Training Time\s*:\s*([\d\.]+)s", log_content)
        if m_time:
            total_time_s = float(m_time.group(1))
        else:
            m_finish = re.search(r"Training finished [a-zA-Z\s]*in\s*([\d\.]+)s", log_content)
            if m_finish:
                total_time_s = float(m_finish.group(1))

        # Test accuracy
        m_test = re.search(r"Final Test Evaluation -> Loss: [\d\.]+, Accuracy:\s*([\d\.]+)%", log_content)
        if m_test:
            final_test_acc = float(m_test.group(1))

        # Best val accuracy
        m_val = re.findall(r"Val Top-1 Acc:\s*([\d\.]+)%", log_content)
        if m_val:
            best_val_acc = max([float(v) for v in m_val])

    # 2. Parse train.csv if available
    if os.path.exists(train_csv):
        with open(train_csv, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            tputs = []
            for row in reader:
                epochs_completed += 1
                try:
                    if "samples_per_sec" in row:
                        tputs.append(float(row["samples_per_sec"]))
                    if best_val_acc is None and "val_accuracy" in row:
                        val_acc = float(row["val_accuracy"]) * 100.0
                        best_val_acc = max(best_val_acc or 0.0, val_acc)
                except Exception:
                    pass
            if tputs:
                avg_tput = sum(tputs) / len(tputs)

    # Comm rounds fallback
    if comm_rounds is None:
        if sync_mode == "gradient_allreduce" or "allreduce" in name:
            comm_rounds = epochs_completed * 150
        elif H_val.isdigit() and int(H_val) > 0:
            comm_rounds = epochs_completed * int(150 / int(H_val))

    # Comm MB
    comm_mb = (comm_rounds * 10.28) if comm_rounds is not None else 0.0

    return {
        "dir": exp_dir,
        "name": name,
        "sync_mode": sync_mode,
        "H": H_val if sync_mode != "gradient_allreduce" else "1 (grad)",
        "policy": policy,
        "epochs": epochs_completed,
        "time_s": total_time_s,
        "comm_rounds": comm_rounds,
        "comm_mb": comm_mb,
        "avg_tput": avg_tput,
        "best_val_acc": best_val_acc,
        "test_acc": final_test_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="HeteroViT-MPI Benchmark Comparison")
    parser.add_argument("--results-dir", default="results", help="Path to results directory")
    parser.add_argument("--pattern", default="*", help="Filter pattern for directories")
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results_dir)
    dirs = sorted(glob.glob(os.path.join(results_dir, args.pattern)), key=os.path.getmtime)

    records = []
    for d in dirs:
        if os.path.isdir(d):
            rec = parse_experiment_dir(d)
            if rec and rec["epochs"] > 0:
                records.append(rec)

    if not records:
        print(f"No completed experiment records found in {results_dir}")
        return

    # Find baseline if any for speedup
    baseline_time = None
    for r in records:
        if "allreduce" in r["name"] and r["time_s"]:
            baseline_time = r["time_s"]
            break

    print("=" * 120)
    print(f" HeteroViT-MPI: Multi-Experiment Benchmark Summary ({len(records)} runs found)")
    print("=" * 120)
    header = (
        f"{'Experiment':<32} {'Mode / H':<15} {'Epochs':<7} {'Time (s)':<10} "
        f"{'Speedup':<9} {'Tput (img/s)':<13} {'Comm Rounds':<12} {'Comm (MB)':<11} "
        f"{'Best Val%':<10} {'Test Acc%':<10}"
    )
    print(header)
    print("-" * 120)

    for r in records:
        time_str = f"{r['time_s']:.1f}s" if r["time_s"] else "N/A"
        speedup_str = "1.00x" if baseline_time and r["time_s"] == baseline_time else (
            f"{baseline_time / r['time_s']:.2f}x" if baseline_time and r["time_s"] else "N/A"
        )
        tput_str = f"{r['avg_tput']:.1f}" if r["avg_tput"] else "N/A"
        val_str = f"{r['best_val_acc']:.2f}%" if r["best_val_acc"] is not None else "N/A"
        test_str = f"{r['test_acc']:.2f}%" if r["test_acc"] is not None else "N/A"
        comm_rnd_str = str(r["comm_rounds"]) if r["comm_rounds"] is not None else "N/A"
        comm_mb_str = f"{r['comm_mb']:.1f}" if r["comm_mb"] else "N/A"
        mode_str = f"{r['sync_mode']} (H={r['H']})" if r['sync_mode'] != 'gradient_allreduce' else "AllReduce"

        print(
            f"{r['name']:<32} {mode_str:<15} {r['epochs']:<7} {time_str:<10} "
            f"{speedup_str:<9} {tput_str:<13} {comm_rnd_str:<12} {comm_mb_str:<11} "
            f"{val_str:<10} {test_str:<10}"
        )
    print("=" * 120)


if __name__ == "__main__":
    main()

