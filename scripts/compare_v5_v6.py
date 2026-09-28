#!/usr/bin/env python3
"""Compare static V5 and dynamic V6 runs using identical post-warmup metrics."""
import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def summarize_run(run_dir: str, warmup_steps: int) -> Dict[str, Any]:
    step_path = os.path.join(run_dir, "step_metrics.csv")
    train_path = os.path.join(run_dir, "train.csv")
    with open(step_path, newline="", encoding="utf-8") as handle:
        step_rows = list(csv.DictReader(handle))
    with open(train_path, newline="", encoding="utf-8") as handle:
        epoch_rows = list(csv.DictReader(handle))

    measured_rows = step_rows[max(0, int(warmup_steps)):]
    gpu_times = [float(row["gpu_step_time_ms"]) for row in measured_rows]
    cpu_times = [float(row["cpu_step_time_ms"]) for row in measured_rows]
    idle_gaps = [abs(gpu - cpu) for gpu, cpu in zip(gpu_times, cpu_times)]
    critical_times = [max(gpu, cpu) for gpu, cpu in zip(gpu_times, cpu_times)]
    val_accuracies = [float(row["val_accuracy"]) for row in epoch_rows]

    summary: Dict[str, Any] = {
        "run_dir": os.path.abspath(run_dir),
        "measured_steps": len(measured_rows),
        "avg_epoch_time_s": _mean([float(row["epoch_time"]) for row in epoch_rows]),
        "avg_throughput_img_s": _mean(
            [float(row["samples_per_sec"]) for row in epoch_rows]
        ),
        "mean_gpu_time_ms": _mean(gpu_times),
        "mean_cpu_time_ms": _mean(cpu_times),
        "mean_critical_time_ms": _mean(critical_times),
        "mean_abs_gpu_cpu_delta_ms": _mean(idle_gaps),
        "p50_idle_gap_ms": _percentile(idle_gaps, 50.0),
        "p95_idle_gap_ms": _percentile(idle_gaps, 95.0),
        "max_idle_gap_ms": max(idle_gaps) if idle_gaps else 0.0,
        "best_validation_accuracy": max(val_accuracies) if val_accuracies else 0.0,
        "final_validation_accuracy": val_accuracies[-1] if val_accuracies else 0.0,
    }

    scheduler_path = os.path.join(run_dir, "v6_scheduler.csv")
    if os.path.exists(scheduler_path):
        with open(scheduler_path, newline="", encoding="utf-8") as handle:
            decisions = list(csv.DictReader(handle))
        summary["scheduler_decisions"] = len(decisions)
        summary["actual_switches"] = sum(
            row.get("action") == "SWITCH" for row in decisions
        )
    else:
        summary["scheduler_decisions"] = 0
        summary["actual_switches"] = 0
    return summary


def improvement_pct(v5_value: float, v6_value: float) -> float:
    return (v5_value - v6_value) / v5_value * 100.0 if v5_value else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-run", required=True, help="Static V5 run directory")
    parser.add_argument("--v6-run", required=True, help="Dynamic V6 run directory")
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    v5 = summarize_run(args.v5_run, args.warmup_steps)
    v6 = summarize_run(args.v6_run, args.warmup_steps)
    comparison = {
        "warmup_steps_excluded": max(0, args.warmup_steps),
        "v5": v5,
        "v6": v6,
        "critical_time_reduction_pct": improvement_pct(
            v5["mean_critical_time_ms"], v6["mean_critical_time_ms"]
        ),
        "idle_gap_reduction_pct": improvement_pct(
            v5["mean_abs_gpu_cpu_delta_ms"], v6["mean_abs_gpu_cpu_delta_ms"]
        ),
        "throughput_change_pct": (
            (v6["avg_throughput_img_s"] - v5["avg_throughput_img_s"])
            / v5["avg_throughput_img_s"]
            * 100.0
            if v5["avg_throughput_img_s"]
            else 0.0
        ),
    }

    output_path = os.path.join(args.v6_run, "v5_vs_v6_comparison.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(comparison, handle, indent=2, sort_keys=True)

    print("V5 vs V6 (positive reduction means V6 improved)")
    print(
        f"  mean critical: {v5['mean_critical_time_ms']:.2f} -> "
        f"{v6['mean_critical_time_ms']:.2f} ms "
        f"({comparison['critical_time_reduction_pct']:+.2f}%)"
    )
    print(
        f"  mean |GPU-CPU|: {v5['mean_abs_gpu_cpu_delta_ms']:.2f} -> "
        f"{v6['mean_abs_gpu_cpu_delta_ms']:.2f} ms "
        f"({comparison['idle_gap_reduction_pct']:+.2f}%)"
    )
    print(
        f"  throughput: {v5['avg_throughput_img_s']:.2f} -> "
        f"{v6['avg_throughput_img_s']:.2f} img/s "
        f"({comparison['throughput_change_pct']:+.2f}%)"
    )
    print(f"  decisions/switches: {v6['scheduler_decisions']}/{v6['actual_switches']}")
    print(f"  report: {output_path}")


if __name__ == "__main__":
    main()
