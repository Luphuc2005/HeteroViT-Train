#!/usr/bin/env python3
"""
Compares Local SGD H=24 and H=40 across seeds (Seed 42 vs Seed 43).
Reports:
- Avg Train Time / Epoch
- Effective Throughput (entire run)
- Total Wall-Clock Time
- Comm Rounds
- Avg Sync Time (ms)
- Best Val Acc (%)
- Best Epoch
- Test Acc (%) from best.weights.h5
"""

import os
import glob
import csv
import re
from typing import Dict, Any

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))

def analyze_run(run_dir: str) -> Dict[str, Any]:
    name = os.path.basename(run_dir)
    train_csv = os.path.join(run_dir, "train.csv")
    run_log = os.path.join(run_dir, "run.log")
    timeline_csv = os.path.join(run_dir, "local_sgd_timeline.csv")
    config_yaml = os.path.join(run_dir, "config.yaml")

    # Extract H and Seed
    H = 0
    seed = 42
    if os.path.exists(config_yaml):
        with open(config_yaml) as f:
            c = f.read()
            m_h = re.search(r"local_sgd_h:\s*(\d+)", c)
            if m_h: H = int(m_h.group(1))
            m_s = re.search(r"seed:\s*(\d+)", c)
            if m_s: seed = int(m_s.group(1))

    # Parse train.csv
    avg_train_time = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs = 20
    if os.path.exists(train_csv):
        with open(train_csv) as f:
            rows = list(csv.DictReader(f))
            epochs = len(rows)
            train_times = [float(r["train_time"]) for r in rows if "train_time" in r]
            avg_train_time = sum(train_times) / len(train_times) if train_times else 0.0
            for r in rows:
                v = float(r.get("val_accuracy", 0)) * 100.0
                ep = int(r.get("epoch", 0))
                if v > best_val_acc:
                    best_val_acc = v
                    best_epoch = ep

    # Parse run.log
    wall_clock = 0.0
    test_acc = 0.0
    comm_rounds = 0
    if os.path.exists(run_log):
        with open(run_log, encoding="utf-8", errors="ignore") as f:
            c = f.read()
            m_time = re.search(r"Training finished.*?in\s+([\d\.]+)s", c)
            if m_time: wall_clock = float(m_time.group(1))
            m_test = re.search(r"Accuracy:\s*([\d\.]+)%", c)
            if m_test: test_acc = float(m_test.group(1))
            m_comm = re.search(r"Total Comm Rounds:\s*(\d+)", c)
            if m_comm: comm_rounds = int(m_comm.group(1))

    # Parse local_sgd_timeline.csv
    avg_sync_ms = 0.0
    if os.path.exists(timeline_csv):
        with open(timeline_csv) as f:
            t_rows = list(csv.DictReader(f))
            sync_times = [float(r["model_sync_ms"]) for r in t_rows if "model_sync_ms" in r]
            avg_sync_ms = sum(sync_times) / len(sync_times) if sync_times else 0.0

    eff_tput = (45000.0 * epochs) / wall_clock if wall_clock > 0 else 0.0
    return {
        "name": name,
        "H": H,
        "seed": seed,
        "epochs": epochs,
        "avg_train_time": avg_train_time,
        "eff_tput": eff_tput,
        "wall_clock": wall_clock,
        "comm_rounds": comm_rounds,
        "avg_sync_ms": avg_sync_ms,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
    }

def main():
    dirs = glob.glob(os.path.join(RESULTS_DIR, "local_sgd_h24_*")) + \
           glob.glob(os.path.join(RESULTS_DIR, "local_sgd_h40_*"))
    dirs = sorted(dirs, key=os.path.getmtime)
    
    runs = []
    for d in dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "train.csv")):
            runs.append(analyze_run(d))

    if not runs:
        print("No H=24 or H=40 runs found in results/.")
        return

    print("=" * 130)
    print(f"{'Experiment Name':<38} {'H':<4} {'Seed':<6} {'Avg Train/Ep':<14} {'Eff Tput':<12} {'Total Time':<12} {'Comm Rnds':<10} {'Sync Time':<12} {'Best Val% (Ep)':<16} {'Test Acc%':<10}")
    print("-" * 130)
    for r in runs:
        m, s = divmod(int(r["wall_clock"]), 60)
        time_str = f"{m}m {s:02d}s" if r["wall_clock"] > 0 else "In Progress"
        best_val_str = f"{r['best_val_acc']:.2f}% (Ep {r['best_epoch']})"
        print(f"{r['name']:<38} {r['H']:<4} {r['seed']:<6} {r['avg_train_time']:<6.2f}s        {r['eff_tput']:<6.1f} img/s  {time_str:<12} {r['comm_rounds']:<10} {r['avg_sync_ms']:<6.1f}ms    {best_val_str:<16} {r['test_acc']:<6.2f}%")
    print("=" * 130)

if __name__ == "__main__":
    main()

