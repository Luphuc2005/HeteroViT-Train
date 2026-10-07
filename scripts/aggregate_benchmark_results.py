#!/usr/bin/env python3
"""
Aggregates and formats benchmark results comparing Clean AllReduce Baseline
against Local SGD (H in 1, 4, 8, 16) on HeteroViT-MPI.
"""

import os
import glob
import json
import csv
import re
from typing import Dict, Any, List

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def parse_run_dir(run_path: str) -> Dict[str, Any]:
    info = {
        "path": run_path,
        "name": os.path.basename(run_path),
        "mode": "unknown",
        "H": 1,
        "policy": "allreduce",
        "epochs": 0,
        "wall_clock_s": 0.0,
        "best_val_acc": 0.0,
        "test_acc": 0.0,
        "tput": 0.0,
        "comm_rounds": 0,
        "comm_bytes_mb": 0.0,
    }

    # 1. Parse config.yaml
    cfg_file = os.path.join(run_path, "config.yaml")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r") as f:
                c_text = f.read()
                m_mode = re.search(r"sync_mode:\s*(\w+)", c_text)
                if m_mode:
                    info["mode"] = m_mode.group(1)
                m_h = re.search(r"local_sgd_h:\s*(\d+)", c_text)
                if m_h:
                    info["H"] = int(m_h.group(1))
                m_ep = re.search(r"epochs:\s*(\d+)", c_text)
                if m_ep:
                    info["epochs"] = int(m_ep.group(1))
                m_pol = re.search(r"avg_policy:\s*(\w+)", c_text)
                if m_pol:
                    info["policy"] = m_pol.group(1)
        except Exception:
            pass

    # 2. Parse train.csv for best val acc and epochs
    train_csv = os.path.join(run_path, "train.csv")
    if os.path.exists(train_csv):
        try:
            with open(train_csv, "r") as f:
                reader = csv.DictReader(f)
                val_accs = []
                for row in reader:
                    val_accs.append(float(row.get("val_accuracy", 0.0)))
                    info["epochs"] = int(row.get("epoch", info["epochs"]))
                    info["tput"] = float(row.get("samples_per_sec", info["tput"]))
                if val_accs:
                    info["best_val_acc"] = max(val_accs) * 100.0
        except Exception:
            pass

    # 3. Parse run.log for total wall-clock, test acc, comm rounds
    log_file = os.path.join(run_path, "run.log")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                # Wall-clock
                m_time = re.search(r"Training finished.*?in\s+([\d\.]+)s", content)
                if m_time:
                    info["wall_clock_s"] = float(m_time.group(1))

                # Test acc
                m_test = re.search(r"Accuracy:\s*([\d\.]+)%", content)
                if m_test:
                    info["test_acc"] = float(m_test.group(1))

                # Comm rounds
                m_rounds = re.search(r"Total Comm Rounds:\s*(\d+)", content)
                if m_rounds:
                    info["comm_rounds"] = int(m_rounds.group(1))
                else:
                    info["comm_rounds"] = info["epochs"] * 150
        except Exception:
            pass

    if info["comm_rounds"] == 0 and info["epochs"] > 0:
        info["comm_rounds"] = info["epochs"] * 150

    # Comm MB
    info["comm_bytes_mb"] = info["comm_rounds"] * 10.28
    return info

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate HeteroViT-MPI Benchmark Results")
    parser.add_argument("--all", action="store_true", help="Include archived verification runs")
    args = parser.parse_args()

    candidate_dirs = [RESULTS_DIR]
    if args.all:
        candidate_dirs.append(os.path.join(RESULTS_DIR, "archive", "audit_verification_1epoch"))

    runs = []
    for d in candidate_dirs:
        if os.path.exists(d):
            runs.extend(glob.glob(os.path.join(d, "*")))
    runs = sorted(runs, key=os.path.getmtime)

    data = []
    for r in runs:
        if os.path.isdir(r) and (os.path.exists(os.path.join(r, "run.log")) or os.path.exists(os.path.join(r, "train.csv"))):
            name = os.path.basename(r)
            if any(k in name for k in ["clean_", "local_sgd_", "allreduce_baseline", "master_agg"]):
                data.append(parse_run_dir(r))

    if not data:
        print("No benchmark runs found in results/.")
        return

    print("=" * 110)
    print(f"{'Experiment Name':<35} {'Mode/H':<15} {'Epochs':<8} {'Wall-Clock':<12} {'Tput (img/s)':<14} {'Best Val%':<10} {'Test%':<8} {'Comm Rounds':<12}")
    print("-" * 110)
    for d in data:
        if "local_sgd" in d['mode']:
            mode_str = f"H={d['H']} (unif)" if d['policy'] == "uniform" else f"H={d['H']}"
        else:
            mode_str = "AllReduce"
        m, s = divmod(int(d["wall_clock_s"]), 60)
        time_str = f"{m}m {s}s" if d["wall_clock_s"] > 0 else "In Progress"
        print(f"{d['name']:<35} {mode_str:<15} {d['epochs']:<8} {time_str:<12} {d['tput']:<14.1f} {d['best_val_acc']:<10.2f} {d['test_acc']:<8.2f} {d['comm_rounds']:<12}")
    print("=" * 110)

if __name__ == "__main__":
    main()
