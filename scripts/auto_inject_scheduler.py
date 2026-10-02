#!/usr/bin/env python3
"""
Auto-Inject Scheduler for HeteroViT-MPI Cluster.

Watches the active training log and automatically injects CPU slowdown 
on a target worker node (e.g. lab03) when specific epochs start.

Usage:
    python3 HeteroViT-MPI/scripts/auto_inject_scheduler.py --epochs 5 15 --node lab03 --duration 300 &
"""

import os
import sys
import time
import re
import glob
import argparse
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INJECT_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "inject_slowdown.sh")


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [AUTO-INJECT] {msg}", flush=True)


def find_active_log_file() -> str:
    """Finds the most recently modified run.log or train.log."""
    results_pattern = os.path.join(PROJECT_ROOT, "results", "dynamic_rebalance_5nodes_*", "run.log")
    logs = glob.glob(results_pattern)
    if logs:
        logs.sort(key=os.path.getmtime, reverse=True)
        return logs[0]
    
    root_log = os.path.join(os.path.dirname(PROJECT_ROOT), "train.log")
    if os.path.exists(root_log):
        return root_log
        
    return ""


def run_injection(node: str, duration: int):
    log(f"⚡ TRIGGERED: Injecting {duration}s slowdown on '{node}'...")
    try:
        res = subprocess.run(
            ["bash", INJECT_SCRIPT, "start", node, str(duration)],
            capture_output=True,
            text=True,
            check=True
        )
        for line in res.stdout.strip().splitlines():
            log(f"   {line}")
    except Exception as e:
        log(f"❌ Failed to run injection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Auto inject slowdown at specific epochs")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 15], help="Target epochs to inject slowdown (e.g. 5 15)")
    parser.add_argument("--node", type=str, default="lab03", help="Target node (default: lab03)")
    parser.add_argument("--duration", type=int, default=300, help="Slowdown duration in seconds (default: 300)")
    parser.add_argument("--log-file", type=str, default="", help="Custom log file to monitor (optional)")
    args = parser.parse_args()

    target_epochs = sorted(list(set(args.epochs)))
    triggered = set()

    log("=" * 70)
    log(" HeteroViT Automated Perturbation Watchdog Started")
    log(f" Target Node     : {args.node}")
    log(f" Target Epochs   : {target_epochs}")
    log(f" Duration/event  : {args.duration}s")
    log("=" * 70)

    log_file = args.log_file or find_active_log_file()
    while not log_file or not os.path.exists(log_file):
        log("Waiting for training log file to appear...")
        time.sleep(3)
        log_file = args.log_file or find_active_log_file()

    log(f"Monitoring log: {log_file}")

    # Regex patterns
    # 1. Start of epoch in step logging: Epoch [05/30]
    epoch_step_re = re.compile(r"Epoch\s*\[(\d+)/\d+\]")
    # 2. End of previous epoch summary: [EPOCH 004/030 SUMMARY]
    summary_re = re.compile(r"\[EPOCH\s*0*(\d+)/\d+\s*SUMMARY\]")

    # Start tailing from current end of file
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        # Seek near end to catch immediately
        f.seek(0, os.SEEK_END)

        while len(triggered) < len(target_epochs):
            line = f.readline()
            if not line:
                time.sleep(1)
                continue

            current_epoch = None

            # Check summary of previous epoch -> immediately triggers next epoch
            m_sum = summary_re.search(line)
            if m_sum:
                prev_ep = int(m_sum.group(1))
                next_ep = prev_ep + 1
                if next_ep in target_epochs and next_ep not in triggered:
                    current_epoch = next_ep

            # Check in-epoch progress log
            m_step = epoch_step_re.search(line)
            if m_step:
                ep = int(m_step.group(1))
                if ep in target_epochs and ep not in triggered:
                    current_epoch = ep

            if current_epoch is not None and current_epoch not in triggered:
                log(f"🎯 Detected start of Epoch {current_epoch}!")
                run_injection(args.node, args.duration)
                triggered.add(current_epoch)
                remaining = [e for e in target_epochs if e not in triggered]
                if remaining:
                    log(f"Waiting for next target epochs: {remaining}...")
                else:
                    log("🎉 All target epoch slowdowns have been successfully scheduled!")

    log("Watchdog task completed. Exiting.")


if __name__ == "__main__":
    main()
