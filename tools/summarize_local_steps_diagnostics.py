#!/usr/bin/env python3
"""Summarize short local-step diagnostic runs and compare one-step weights."""

import argparse
import csv
import math
import os
import re
from datetime import datetime


def parse_case(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must use LABEL=RUN_DIR")
    label, run_dir = value.split("=", 1)
    return label, os.path.abspath(run_dir)


def read_last_metrics(run_dir):
    csv_path = os.path.join(run_dir, "train.csv")
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("no metric rows in {}".format(csv_path))

    row = rows[-1]
    log_path = os.path.join(run_dir, "run.log")
    max_steps = "?"
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            match = re.search(r"Reached max_steps=(\d+)", handle.read())
        if match:
            max_steps = match.group(1)

    return {
        "steps": max_steps,
        "train_loss": float(row["train_loss"]),
        "train_accuracy": float(row["train_accuracy"]),
        "val_loss": float(row["val_loss"]),
        "val_accuracy": float(row["val_accuracy"]),
        "train_time": float(row["train_time"]),
        "samples_per_sec": float(row["samples_per_sec"]),
    }


def read_h5_datasets(path):
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("h5py and numpy are required for weight comparison") from exc

    datasets = {}
    with h5py.File(path, "r") as handle:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = np.asarray(obj[...])
        handle.visititems(visitor)
    return datasets


def compare_weights(left_path, right_path):
    import numpy as np

    left = read_h5_datasets(left_path)
    right = read_h5_datasets(right_path)
    common = sorted(set(left) & set(right))
    mismatched = sorted(set(left) ^ set(right))
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    compared = 0

    for name in common:
        if left[name].shape != right[name].shape:
            mismatched.append(name + " (shape)")
            continue
        lhs = left[name].astype(np.float64, copy=False)
        rhs = right[name].astype(np.float64, copy=False)
        diff = lhs - rhs
        if diff.size:
            max_abs = max(max_abs, float(np.max(np.abs(diff))))
        diff_sq += float(np.sum(diff * diff))
        ref_sq += float(np.sum(lhs * lhs))
        compared += int(diff.size)

    l2 = math.sqrt(diff_sq)
    relative_l2 = l2 / max(math.sqrt(ref_sq), 1e-30)
    return {
        "datasets": len(common),
        "values": compared,
        "mismatched": mismatched,
        "max_abs": max_abs,
        "l2": l2,
        "relative_l2": relative_l2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--compare", nargs=2, metavar=("LEFT_LABEL", "RIGHT_LABEL"))
    args = parser.parse_args()

    cases = dict(args.case)
    metrics = {label: read_last_metrics(path) for label, path in args.case}

    comparison = None
    if args.compare:
        left_label, right_label = args.compare
        left_weights = os.path.join(cases[left_label], "checkpoints", "last.weights.h5")
        right_weights = os.path.join(cases[right_label], "checkpoints", "last.weights.h5")
        comparison = compare_weights(left_weights, right_weights)

    lines = [
        "# Local-steps diagnostic summary",
        "",
        "Generated: `{}`".format(datetime.now().isoformat(timespec="seconds")),
        "",
        "## Short-run metrics",
        "",
        "| Case | Steps | Train loss | Train acc | Val loss | Val acc | Train time (s) | Samples/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _ in args.case:
        item = metrics[label]
        lines.append(
            "| {} | {} | {:.5f} | {:.2f}% | {:.5f} | {:.2f}% | {:.2f} | {:.2f} |".format(
                label,
                item["steps"],
                item["train_loss"],
                item["train_accuracy"] * 100.0,
                item["val_loss"],
                item["val_accuracy"] * 100.0,
                item["train_time"],
                item["samples_per_sec"],
            )
        )

    if comparison is not None:
        left_label, right_label = args.compare
        lines.extend([
            "",
            "## One-step weight comparison",
            "",
            "Comparison: `{}` vs `{}`.".format(left_label, right_label),
            "",
            "- Compared datasets: `{}`".format(comparison["datasets"]),
            "- Compared scalar values: `{}`".format(comparison["values"]),
            "- Maximum absolute difference: `{:.8e}`".format(comparison["max_abs"]),
            "- L2 difference: `{:.8e}`".format(comparison["l2"]),
            "- Relative L2 difference: `{:.8e}`".format(comparison["relative_l2"]),
            "- Dataset-name/shape mismatches: `{}`".format(len(comparison["mismatched"])),
            "",
        ])
        if comparison["max_abs"] <= 1e-5 and not comparison["mismatched"]:
            lines.append("Result: weights are numerically close at tolerance `1e-5`.")
        else:
            lines.append(
                "Result: K=1 does not reproduce the baseline one-step weights at tolerance `1e-5`."
            )

    lines.extend([
        "",
        "## Run directories",
        "",
    ])
    for label, path in args.case:
        lines.append("- `{}`: `{}`".format(label, path))

    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(args.output)


if __name__ == "__main__":
    main()
