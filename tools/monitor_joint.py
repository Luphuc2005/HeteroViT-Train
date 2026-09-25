#!/usr/bin/env python3
"""Live Monitor for Joint Heterogeneous Training Benchmarks."""
import os
import sys
import time
import glob
import csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_latest_run_dir():
    patterns = [
        os.path.join(PROJECT_ROOT, "results", "joint_benchmarks", "*", "*"),
        os.path.join(PROJECT_ROOT, "results", "joint_benchmarks", "*"),
    ]
    all_dirs = []
    for pat in patterns:
        for d in glob.glob(pat):
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "train.csv")):
                all_dirs.append(d)
    if not all_dirs:
        return None
    return max(all_dirs, key=os.path.getmtime)

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else get_latest_run_dir()
    if not target_dir or not os.path.exists(target_dir):
        print("No active or completed joint run found.")
        return

    print(f"Monitoring: {target_dir}")
    train_csv = os.path.join(target_dir, "train.csv")
    step_csv = os.path.join(target_dir, "step_metrics.csv")

    last_step_count = 0
    last_epoch_count = 0

    while True:
        # Check Epochs
        if os.path.exists(train_csv):
            with open(train_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > last_epoch_count and len(lines) > 1:
                    last_epoch_count = len(lines)
                    last_line = lines[-1].strip().split(",")
                    print(
                        f"\n[EPOCH {int(last_line[0]):02d}] "
                        f"Time: {float(last_line[5]):4.1f}s | "
                        f"Throughput: {float(last_line[6]):6.1f} img/s | "
                        f"GPU: {float(last_line[7]):5.1f}ms | "
                        f"CPU: {float(last_line[8]):5.1f}ms | "
                        f"Wait: {float(last_line[9]):4.1f}ms | "
                        f"Val Acc: {float(last_line[4])*100:5.2f}%"
                    )

        # Check Latest Step
        if os.path.exists(step_csv):
            with open(step_csv, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last = lines[-1].strip().split(",")
                    cur_step_count = len(lines)
                    if cur_step_count != last_step_count and int(last[1]) % 10 == 0:
                        last_step_count = cur_step_count
                        sys.stdout.write(
                            f"\r  --> E{int(last[0]):02d} S{int(last[1]):03d} | "
                            f"GPU: {float(last[2]):5.1f}ms | "
                            f"CPU: {float(last[3]):5.1f}ms | "
                            f"Wait: {float(last[4]):4.1f}ms | "
                            f"{float(last[7]):6.1f} img/s | "
                            f"Loss: {float(last[8]):.4f} | Acc: {float(last[9]):5.1f}%"
                        )
                        sys.stdout.flush()

        time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited monitor.")
