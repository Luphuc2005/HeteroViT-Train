"""Offline Validation Script for Phase 1 Cost Model.

Runs benchmark candidates on the cluster, gathers step-level instrumentation,
and compiles the Predicted vs Actual comparison table.
"""
import os
import sys
import glob
import json
import csv
import subprocess
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.offline_scheduler import OfflineScheduler

HOSTFILE_5NODES = os.path.abspath(os.path.join(PROJECT_ROOT, "../mpi_cluster/hostfile"))
HOSTFILE_4NODES = os.path.abspath(os.path.join(PROJECT_ROOT, "../mpi_cluster/hostfile_no_lab02"))
PYTHON_BY_HOST = os.path.abspath(os.path.join(PROJECT_ROOT, "../mpi_cluster/python_by_host.sh"))
TRAIN_SCRIPT = os.path.abspath(os.path.join(PROJECT_ROOT, "train_mpi.py"))

CANDIDATES = [
    {
        "name": "C1 (Uniform 5-node)",
        "allocation": {"lab01": 128, "lab02": 128, "lab03": 128, "lab04": 128, "lab05": 128},
        "config_path": os.path.join(PROJECT_ROOT, "configs", "mpi", "val_c1_uniform_5nodes.yaml"),
        "exp_prefix": "val_c1_uniform_5nodes",
        "world_size": 5,
        "hostfile": HOSTFILE_5NODES,
        "steps": 10,
    },
    {
        "name": "C2 (Hetero Static 5-node)",
        "allocation": {"lab01": 384, "lab02": 64, "lab03": 48, "lab04": 48, "lab05": 96},
        "config_path": os.path.join(PROJECT_ROOT, "configs", "mpi", "val_c2_hetero_static_5nodes.yaml"),
        "exp_prefix": "val_c2_hetero_static_5nodes",
        "world_size": 5,
        "hostfile": HOSTFILE_5NODES,
        "steps": 12,
    },
    {
        "name": "C3 (Balanced 5-node)",
        "allocation": {"lab01": 256, "lab02": 16, "lab03": 24, "lab04": 24, "lab05": 48},
        "config_path": os.path.join(PROJECT_ROOT, "configs", "mpi", "val_c3_balanced_5nodes.yaml"),
        "exp_prefix": "val_c3_balanced_5nodes",
        "world_size": 5,
        "hostfile": HOSTFILE_5NODES,
        "steps": 12,
    },
    {
        "name": "C4 (4-node Drop lab02)",
        "allocation": {"lab01": 256, "lab02": 0, "lab03": 32, "lab04": 32, "lab05": 64},
        "config_path": os.path.join(PROJECT_ROOT, "configs", "mpi", "val_c4_drop_lab02_4nodes.yaml"),
        "exp_prefix": "val_c4_drop_lab02_4nodes",
        "world_size": 4,
        "hostfile": HOSTFILE_4NODES,
        "steps": 12,
    },
]


def run_candidate_benchmark(cand: dict) -> str:
    """Executes fixed benchmark run via mpirun and returns results dir."""
    np_val = cand["world_size"]
    hfile = cand["hostfile"]
    steps = cand["steps"]
    config_path = cand["config_path"]

    cmd = [
        "mpirun",
        "-x", "PYTHONNOUSERSITE=1",
        "--mca", "btl_tcp_if_include", "192.168.1.0/24",
        "--mca", "oob_tcp_if_include", "192.168.1.0/24",
        "-np", str(np_val),
        "--hostfile", hfile,
        "--map-by", "ppr:1:node",
        PYTHON_BY_HOST,
        TRAIN_SCRIPT,
        "--config", config_path,
        "--max-steps", str(steps),
    ]

    print(f"\n[*] Executing candidate: {cand['name']} (steps={steps})...", flush=True)
    subprocess.run(cmd, check=True)

    # Find the latest results directory
    results_base = os.path.join(PROJECT_ROOT, "results")
    exp_prefix = cand["exp_prefix"]
    subdirs = [
        os.path.join(results_base, d) for d in os.listdir(results_base)
        if os.path.isdir(os.path.join(results_base, d)) and d.startswith(exp_prefix)
    ]
    subdirs.sort(key=os.path.getmtime, reverse=True)
    return subdirs[0] if subdirs else ""


def parse_actual_metrics(results_dir: str, warmup_steps: int = 2) -> dict:
    """Parses step_metrics.csv or run.log to compute actual critical path step time and throughput."""
    if not results_dir or not os.path.exists(results_dir):
        return {"actual_t_crit_ms": 0.0}

    # 1. Try step_metrics.csv first (from RuntimeProfiler)
    csv_path = os.path.join(results_dir, "step_metrics.csv")
    if os.path.exists(csv_path):
        step_times = []
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s_id = int(row.get("step_id", 0))
                    if s_id > warmup_steps:
                        step_times.append(float(row["step_ms"]))
            if step_times:
                return {"actual_t_crit_ms": float(np.mean(step_times))}
        except Exception as e:
            print(f"[Warning] Failed parsing {csv_path}: {e}")

    # 2. Fallback to run.log regex parsing
    log_path = os.path.join(results_dir, "run.log")
    if os.path.exists(log_path):
        import re
        step_times = []
        pattern = re.compile(r"Step:\s*([0-9.]+)ms")
        step_count = 0
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    step_count += 1
                    if step_count > warmup_steps:
                        step_times.append(float(m.group(1)))
        if not step_times and step_count > 0:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = pattern.search(line)
                    if m:
                        step_times.append(float(m.group(1)))
        if step_times:
            return {"actual_t_crit_ms": float(np.mean(step_times))}

    return {"actual_t_crit_ms": 0.0}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-only", action="store_true", help="Parse existing results without re-running cluster")
    args = parser.parse_args()

    scheduler = OfflineScheduler()

    validation_rows = []

    print("=" * 120)
    print(" HETEROVIT-MPI: OFFLINE VALIDATION (PHASE 1 COST MODEL)")
    print("=" * 120)

    for cand in CANDIDATES:
        # 1. Cost model predictions
        ev = scheduler.evaluate(cand["allocation"])
        pred_t_crit = ev.t_critical_ms
        pred_tput = ev.predicted_throughput
        total_batch = ev.total_batch

        # 2. Run actual benchmark on cluster (or find existing latest run)
        if args.parse_only:
            results_base = os.path.join(PROJECT_ROOT, "results")
            exp_prefix = cand["exp_prefix"]
            subdirs = [
                os.path.join(results_base, d) for d in os.listdir(results_base)
                if os.path.isdir(os.path.join(results_base, d)) and d.startswith(exp_prefix)
            ]
            subdirs.sort(key=os.path.getmtime, reverse=True)
            res_dir = subdirs[0] if subdirs else ""
        else:
            res_dir = run_candidate_benchmark(cand)

        # 3. Parse actual measurements
        act_info = parse_actual_metrics(res_dir, warmup_steps=2)
        act_t_crit = act_info["actual_t_crit_ms"]
        act_tput = float(total_batch) / (act_t_crit / 1000.0) if act_t_crit > 0 else 0.0

        # 4. Compute error %
        error_pct = (abs(act_t_crit - pred_t_crit) / pred_t_crit) * 100.0 if pred_t_crit > 0 else 0.0

        alloc_summary = ", ".join(f"{k}:{v}" for k, v in cand["allocation"].items())

        validation_rows.append({
            "candidate": cand["name"],
            "allocation": alloc_summary,
            "total_batch": total_batch,
            "pred_t_crit": pred_t_crit,
            "act_t_crit": act_t_crit,
            "pred_tput": pred_tput,
            "act_tput": act_tput,
            "error_pct": error_pct,
        })

    # Print Comparison Table
    print("\n" + "=" * 120)
    print(" FINAL OFFLINE VALIDATION TABLE: PREDICTED VS ACTUAL")
    print("=" * 120)
    print(f"{'Candidate':<24} | {'Node Allocation':<36} | {'Pred T_crit':<12} | {'Actual T_crit':<12} | {'Pred Tput':<10} | {'Act Tput':<10} | {'Error %':<8}")
    print("-" * 120)
    for r in validation_rows:
        print(
            f"{r['candidate']:<24} | {r['allocation']:<36} | "
            f"{r['pred_t_crit']:9.2f} ms | {r['act_t_crit']:9.2f} ms | "
            f"{r['pred_tput']:7.1f} i/s | {r['act_tput']:7.1f} i/s | {r['error_pct']:6.2f}%"
        )
    print("=" * 120)

    # Save to JSON
    out_table_path = os.path.join(PROJECT_ROOT, "profiles", "validation_results.json")
    with open(out_table_path, "w", encoding="utf-8") as f:
        json.dump(validation_rows, f, indent=2)
    print(f"[SUCCESS] Validation results saved to: {out_table_path}")


if __name__ == "__main__":
    main()
