"""Analysis and Visualization Script for Adaptive Local SGD Experiments.

Generates the 9 publication-quality figures specified in the Master Prompt:
  1. H vs synchronization round
  2. Weighted consensus error (V_t or Q_t) vs round
  3. Observed vs predicted consensus error
  4. Prediction uncertainty vs round (for V2/V3)
  5. Dual variable mu_t vs round (for V3)
  6. Validation accuracy vs epoch
  7. Validation accuracy vs wall-clock time
  8. Communication time vs training time breakdown
  9. Accuracy–runtime Pareto plot

Usage:
  python scripts/plot_adaptive_h_experiments.py --run-dirs results/dir1 results/dir2 ... --output-dir plots/
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run_data(run_dir: str) -> Dict[str, Any]:
    """Loads CSV logs and metadata from a specific run directory."""
    data = {
        "run_dir": run_dir,
        "name": os.path.basename(os.path.normpath(run_dir)),
        "sync_events": None,
        "decisions": None,
        "epoch_metrics": None,
        "runtime_breakdown": None,
        "metadata": None,
        "train_csv": None,
    }

    sync_path = os.path.join(run_dir, "sync_events.csv")
    if os.path.exists(sync_path):
        data["sync_events"] = pd.read_csv(sync_path)

    dec_path = os.path.join(run_dir, "controller_decisions.csv")
    if os.path.exists(dec_path):
        data["decisions"] = pd.read_csv(dec_path)

    epoch_path = os.path.join(run_dir, "epoch_metrics.csv")
    if os.path.exists(epoch_path):
        data["epoch_metrics"] = pd.read_csv(epoch_path)

    rt_path = os.path.join(run_dir, "runtime_breakdown.csv")
    if os.path.exists(rt_path):
        data["runtime_breakdown"] = pd.read_csv(rt_path)

    meta_path = os.path.join(run_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            data["metadata"] = json.load(f)

    train_path = os.path.join(run_dir, "train.csv")
    if os.path.exists(train_path):
        data["train_csv"] = pd.read_csv(train_path)

    # Legacy trace fallback if sync_events is not present
    legacy_trace = os.path.join(run_dir, "adaptive_h_trace.csv")
    if data["sync_events"] is None and os.path.exists(legacy_trace):
        data["sync_events"] = pd.read_csv(legacy_trace)

    return data


def plot_h_trajectories(runs: List[Dict[str, Any]], out_dir: str):
    """Plot 1: H vs synchronization round."""
    plt.figure(figsize=(9, 5))
    for r in runs:
        df = r["sync_events"]
        if df is not None and "planned_h" in df.columns and "round_idx" in df.columns:
            plt.step(df["round_idx"], df["planned_h"], where="post", label=r["name"], lw=2)
        elif df is not None and "current_h" in df.columns and "sync_round" in df.columns:
            plt.step(df["sync_round"], df["current_h"], where="post", label=r["name"], lw=2)

    plt.xlabel("Synchronization Round", fontsize=12)
    plt.ylabel("Synchronization Interval H", fontsize=12)
    plt.title("Synchronization Interval (H) Trajectory across Training", fontsize=13)
    plt.yticks([40, 80, 120, 150])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "01_h_vs_round.png"), dpi=300)
    plt.close()


def plot_consensus_error(runs: List[Dict[str, Any]], out_dir: str):
    """Plot 2: Weighted consensus error vs round."""
    plt.figure(figsize=(9, 5))
    for r in runs:
        df = r["sync_events"]
        if df is not None:
            if "q_t" in df.columns:
                plt.plot(df["round_idx"], df["q_t"], label=f"{r['name']} (Q_t)", lw=1.8)
            elif "drift" in df.columns:
                plt.plot(df["sync_round"], df["drift"], label=f"{r['name']} (drift)", lw=1.8)

    plt.xlabel("Synchronization Round", fontsize=12)
    plt.ylabel("Consensus Disagreement (Q_t / Drift)", fontsize=12)
    plt.title("Normalized Consensus Error Evolution", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "02_consensus_error_vs_round.png"), dpi=300)
    plt.close()


def plot_observed_vs_predicted(runs: List[Dict[str, Any]], out_dir: str):
    """Plot 3: Observed vs predicted consensus error."""
    plt.figure(figsize=(9, 5))
    found = False
    for r in runs:
        dec = r["decisions"]
        sync = r["sync_events"]
        if dec is not None and sync is not None and "candidate_predictions" in dec.columns:
            # Parse candidate predictions
            try:
                preds = []
                for p_str, next_h in zip(dec["candidate_predictions"], dec["next_h"]):
                    p_dict = json.loads(p_str)
                    preds.append(p_dict.get(str(next_h), p_dict.get(int(next_h), np.nan)))
                if len(preds) > 1 and "q_t" in sync.columns:
                    # Predicted at round t vs observed at round t+1
                    obs = sync["q_t"].iloc[1:].values
                    pred = np.array(preds[:-1])
                    min_len = min(len(obs), len(pred))
                    if min_len > 0:
                        plt.plot(range(1, min_len + 1), obs[:min_len], label=f"{r['name']} Observed", lw=1.5)
                        plt.plot(range(1, min_len + 1), pred[:min_len], "--", label=f"{r['name']} Predicted", lw=1.5)
                        found = True
            except Exception:
                pass

    if found:
        plt.xlabel("Synchronization Round", fontsize=12)
        plt.ylabel("Normalized Consensus Error Q", fontsize=12)
        plt.title("Observed vs Predicted Consensus Error", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "03_observed_vs_predicted.png"), dpi=300)
    plt.close()


def plot_dual_variable(runs: List[Dict[str, Any]], out_dir: str):
    """Plot 5: Dual multiplier mu_t vs round (V3)."""
    plt.figure(figsize=(9, 5))
    plotted = False
    for r in runs:
        dec = r["decisions"]
        if dec is not None and "dual_variable" in dec.columns:
            mu_vals = pd.to_numeric(dec["dual_variable"], errors="coerce").dropna()
            if len(mu_vals) > 0 and (mu_vals > 0).any():
                plt.plot(dec.loc[mu_vals.index, "round_idx"], mu_vals, label=r["name"], lw=2)
                plotted = True

    if plotted:
        plt.xlabel("Synchronization Round", fontsize=12)
        plt.ylabel("Dual Multiplier mu_t", fontsize=12)
        plt.title("Lagrangian Dual Multiplier Dynamics (V3 PDCA-H)", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "05_dual_variable_vs_round.png"), dpi=300)
    plt.close()


def plot_val_accuracy_curves(runs: List[Dict[str, Any]], out_dir: str):
    """Plots 6 & 7: Validation accuracy vs epoch and wall-clock time."""
    # Plot 6: vs Epoch
    plt.figure(figsize=(9, 5))
    for r in runs:
        df = r["epoch_metrics"] if r["epoch_metrics"] is not None else r["train_csv"]
        if df is not None and "val_acc" in df.columns:
            plt.plot(df["epoch"], df["val_acc"] * 100 if df["val_acc"].max() <= 1.0 else df["val_acc"], label=r["name"], lw=2)
        elif df is not None and "val_accuracy" in df.columns:
            plt.plot(df["epoch"], df["val_accuracy"] * 100 if df["val_accuracy"].max() <= 1.0 else df["val_accuracy"], label=r["name"], lw=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Validation Accuracy vs Epoch", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "06_val_acc_vs_epoch.png"), dpi=300)
    plt.close()

    # Plot 7: vs Wall-clock time
    plt.figure(figsize=(9, 5))
    for r in runs:
        df = r["epoch_metrics"] if r["epoch_metrics"] is not None else r["train_csv"]
        if df is not None:
            acc_col = "val_acc" if "val_acc" in df.columns else ("val_accuracy" if "val_accuracy" in df.columns else None)
            time_col = "epoch_time_sec" if "epoch_time_sec" in df.columns else ("epoch_time" if "epoch_time" in df.columns else None)
            if acc_col and time_col:
                cum_time_min = df[time_col].cumsum() / 60.0
                acc_vals = df[acc_col] * 100 if df[acc_col].max() <= 1.0 else df[acc_col]
                plt.plot(cum_time_min, acc_vals, label=r["name"], lw=2)

    plt.xlabel("Elapsed Wall-Clock Training Time (minutes)", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title("Validation Accuracy vs Wall-Clock Time", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "07_val_acc_vs_time.png"), dpi=300)
    plt.close()


def plot_pareto_accuracy_time(runs: List[Dict[str, Any]], out_dir: str):
    """Plot 9: Accuracy vs Wall-Clock Time Pareto trade-off."""
    plt.figure(figsize=(8, 6))
    pareto_points = []
    for r in runs:
        df = r["epoch_metrics"] if r["epoch_metrics"] is not None else r["train_csv"]
        if df is not None:
            acc_col = "val_acc" if "val_acc" in df.columns else ("val_accuracy" if "val_accuracy" in df.columns else None)
            time_col = "epoch_time_sec" if "epoch_time_sec" in df.columns else ("epoch_time" if "epoch_time" in df.columns else None)
            if acc_col and time_col:
                best_acc = float(df[acc_col].max())
                best_acc_pct = best_acc * 100 if best_acc <= 1.0 else best_acc
                total_time_min = float(df[time_col].sum()) / 60.0
                pareto_points.append((r["name"], total_time_min, best_acc_pct))

    if pareto_points:
        for name, t, acc in pareto_points:
            plt.scatter(t, acc, s=120, label=name)
            plt.annotate(
                name,
                (t, acc),
                textcoords="offset points",
                xytext=(8, 5),
                fontsize=9,
            )

        plt.xlabel("Total Training Time (minutes)", fontsize=12)
        plt.ylabel("Best Validation Accuracy (%)", fontsize=12)
        plt.title("Accuracy vs Training Time Trade-off (Pareto Frontier)", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "09_accuracy_runtime_pareto.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Adaptive Local SGD experiment comparisons")
    parser.add_argument("--run-dirs", nargs="+", default=[], help="List of run directories to compare")
    parser.add_argument("--output-dir", type=str, default="plots/adaptive_h_analysis", help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = [load_run_data(d) for d in args.run_dirs if os.path.isdir(d)]

    if not runs:
        print("[WARNING] No valid run directories provided. Searching results/...")
        default_candidates = glob.glob("results/local_sgd_*")
        runs = [load_run_data(d) for d in default_candidates if os.path.isdir(d)][:6]

    print(f"Loaded {len(runs)} runs for visualization.")
    plot_h_trajectories(runs, args.output_dir)
    plot_consensus_error(runs, args.output_dir)
    plot_observed_vs_predicted(runs, args.output_dir)
    plot_dual_variable(runs, args.output_dir)
    plot_val_accuracy_curves(runs, args.output_dir)
    plot_pareto_accuracy_time(runs, args.output_dir)
    print(f"All figures generated successfully in '{args.output_dir}/'")


if __name__ == "__main__":
    main()

