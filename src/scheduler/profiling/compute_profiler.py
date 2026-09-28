"""Compute Profiler for 5-Node Heterogeneous Cluster.

Profiles pure local forward + backward computation time across candidate local batch sizes.
- Warmup: 10 steps (discarded)
- Measurement: 50 stable steps
- Measures:
  - mean_compute_ms
  - std_compute_ms
  - throughput (img/s)
- Saves profile to JSON and CSV in profiles/
"""
import os
import sys
import time
import json
import csv
import socket
import argparse
from typing import Dict, Any, List
import numpy as np

# Early MPI and Device Setup
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
hostname = socket.gethostname()

if rank == 0:
    if "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ["CUDA_VISIBLE_DEVICES"] == "":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tensorflow as tf
from src.utils.config import load_config
from src.models.vit import build_vit_from_config
from src.training.base_trainer import get_optimizer
from src.training.trainer_cpu import configure_cpu_runtime

NODE_NAMES = {
    0: "lab01",
    1: "lab02",
    2: "lab03",
    3: "lab04",
    4: "lab05",
}

DEFAULT_BATCH_CANDIDATES = {
    0: [32, 64, 128, 192, 232, 240, 256, 384],
    1: [8, 16, 24, 32, 48, 64],
    2: [8, 16, 24, 32, 48, 64],
    3: [8, 16, 24, 32, 48, 64],
    4: [16, 32, 48, 64, 96, 128],
}


def build_step_fn(model, loss_fn, optimizer, strategy, local_batch, accum_steps):
    """Builds compiled local gradient computation matching MPITrainer."""
    if strategy is not None:
        if accum_steps > 1:
            def replica_step_fn(images, labels):
                img_chunks = tf.split(images, accum_steps, axis=0)
                lbl_chunks = tf.split(labels, accum_steps, axis=0)
                accum_grads = [tf.zeros_like(v) for v in model.trainable_variables]
                total_loss = tf.constant(0.0, dtype=tf.float32)
                total_acc = tf.constant(0.0, dtype=tf.float32)

                for m_img, m_lbl in zip(img_chunks, lbl_chunks):
                    with tf.GradientTape() as tape:
                        predictions = model(m_img, training=True)
                        per_example_loss = loss_fn(m_lbl, predictions)
                        m_loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=local_batch)
                    grads = tape.gradient(m_loss, model.trainable_variables)
                    accum_grads = [
                        ag + (g if g is not None else tf.zeros_like(ag))
                        for ag, g in zip(accum_grads, grads)
                    ]
                    total_loss += m_loss
                    pred_labels = tf.argmax(predictions, axis=-1, output_type=m_lbl.dtype)
                    m_acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, m_lbl), tf.float32))
                    total_acc += m_acc / float(accum_steps)

                return total_loss, accum_grads, total_acc

            @tf.function
            def compute_grads(images, labels):
                per_replica_losses, per_replica_grads, per_replica_accs = strategy.run(
                    replica_step_fn, args=(images, labels)
                )
                local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                local_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
                local_grads = [
                    strategy.reduce(tf.distribute.ReduceOp.SUM, g, axis=None)
                    for g in per_replica_grads
                ]
                return local_loss, local_acc, local_grads

            return compute_grads
        else:
            def replica_step_fn(images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    per_example_loss = loss_fn(labels, predictions)
                    loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=local_batch)
                grads = tape.gradient(loss, model.trainable_variables)
                pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                return loss, grads, acc

            @tf.function
            def compute_grads(images, labels):
                per_replica_losses, per_replica_grads, per_replica_accs = strategy.run(
                    replica_step_fn, args=(images, labels)
                )
                local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                local_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
                local_grads = [
                    strategy.reduce(tf.distribute.ReduceOp.SUM, g, axis=None)
                    for g in per_replica_grads
                ]
                return local_loss, local_acc, local_grads

            return compute_grads
    else:
        if accum_steps > 1:
            @tf.function
            def compute_grads(images, labels):
                img_chunks = tf.split(images, accum_steps, axis=0)
                lbl_chunks = tf.split(labels, accum_steps, axis=0)
                accum_grads = [tf.zeros_like(v) for v in model.trainable_variables]
                total_loss = tf.constant(0.0, dtype=tf.float32)
                total_acc = tf.constant(0.0, dtype=tf.float32)

                for m_img, m_lbl in zip(img_chunks, lbl_chunks):
                    with tf.GradientTape() as tape:
                        predictions = model(m_img, training=True)
                        loss_m = loss_fn(m_lbl, predictions)
                        scaled_loss_m = loss_m / float(accum_steps)
                    grads = tape.gradient(scaled_loss_m, model.trainable_variables)
                    accum_grads = [
                        ag + (g if g is not None else tf.zeros_like(ag))
                        for ag, g in zip(accum_grads, grads)
                    ]
                    total_loss += scaled_loss_m
                    pred_labels = tf.argmax(predictions, axis=-1, output_type=m_lbl.dtype)
                    m_acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, m_lbl), tf.float32))
                    total_acc += m_acc / float(accum_steps)

                return total_loss, total_acc, accum_grads

            return compute_grads
        else:
            @tf.function
            def compute_grads(images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = loss_fn(labels, predictions)
                grads = tape.gradient(loss, model.trainable_variables)
                pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                return loss, acc, grads

            return compute_grads


def profile_node(config_path: str, warmup_steps: int = 10, bench_steps: int = 50) -> Dict[str, Any]:
    """Runs compute profiling on local node for its configured batch sizes."""
    config = load_config(config_path)
    training_cfg = config.get("training", {})

    # Configure CPU runtime if rank != 0
    if rank != 0 or "cpu" in config:
        configure_cpu_runtime(config.get("cpu", {}))

    physical_gpus = tf.config.list_physical_devices("GPU")
    num_gpus = len(physical_gpus) if rank == 0 else 0

    strategy = None
    if rank == 0 and num_gpus > 0:
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        if num_gpus > 1:
            strategy = tf.distribute.MirroredStrategy()

    device_type = f"{num_gpus}xGPU" if (rank == 0 and num_gpus > 0) else "CPU"
    node_id = NODE_NAMES.get(rank, f"node_{rank}")

    # Build model
    image_size = int(config.get("model", {}).get("image_size", 32))
    if rank == 0 and strategy is not None:
        with strategy.scope():
            model = build_vit_from_config(config)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            optimizer = get_optimizer(training_cfg)
            dummy = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
            _ = model(dummy, training=False)
    else:
        model = build_vit_from_config(config)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.AUTO
        )
        optimizer = get_optimizer(training_cfg)
        dummy = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
        _ = model(dummy, training=False)

    batch_list = DEFAULT_BATCH_CANDIDATES.get(rank, [32, 64])
    print(f"[{node_id} (Rank {rank}, {device_type})] Starting compute profile for batches: {batch_list}", flush=True)

    node_results = {
        "node_id": node_id,
        "rank": rank,
        "host": hostname,
        "device_type": device_type,
        "profiles": {},
    }

    # Micro-batch threshold for gradient accumulation
    micro_batch = int(training_cfg.get("grad_accum", {}).get("micro_batch_size", 128))

    for b in batch_list:
        accum_steps = 1
        if rank == 0 and b > micro_batch:
            accum_steps = b // micro_batch

        step_fn = build_step_fn(model, loss_fn, optimizer, strategy, b, accum_steps)

        # Generate synthetic input batch
        np_x = np.random.randn(b, image_size, image_size, 3).astype(np.float32)
        np_y = np.random.randint(0, 10, size=(b,), dtype=np.int32)
        x_tensor = tf.convert_to_tensor(np_x)
        y_tensor = tf.convert_to_tensor(np_y)

        # Warmup steps
        for _ in range(warmup_steps):
            out = step_fn(x_tensor, y_tensor)
            grads = out[2] if len(out) == 3 else out[1]
            _ = [g.numpy() for g in grads if g is not None]

        # Measurement steps
        durations = []
        for _ in range(bench_steps):
            t0 = time.perf_counter()
            out = step_fn(x_tensor, y_tensor)
            grads = out[2] if len(out) == 3 else out[1]
            _ = [g.numpy() for g in grads if g is not None]
            t1 = time.perf_counter()
            durations.append((t1 - t0) * 1000.0)  # ms

        mean_ms = float(np.mean(durations))
        std_ms = float(np.std(durations))
        tput = float(b) / (mean_ms / 1000.0)

        print(
            f"  [{node_id}] Batch {b:3d} (accum={accum_steps}): "
            f"Mean={mean_ms:7.2f} ms | Std={std_ms:5.2f} ms | Throughput={tput:6.1f} img/s",
            flush=True,
        )

        node_results["profiles"][str(b)] = {
            "batch_size": b,
            "accum_steps": accum_steps,
            "mean_compute_ms": mean_ms,
            "std_compute_ms": std_ms,
            "min_compute_ms": float(np.min(durations)),
            "max_compute_ms": float(np.max(durations)),
            "throughput_img_s": tput,
            "sample_count": bench_steps,
        }

    return node_results


def main():
    parser = argparse.ArgumentParser(description="Distributed Compute Profiler for HeteroViT")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "mpi", "baseline_5nodes.yaml"),
        help="Path to YAML config",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup steps per batch size",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Benchmarking steps per batch size",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(PROJECT_ROOT, "profiles", "compute_profile.json"),
        help="Path to save compute JSON profile",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=os.path.join(PROJECT_ROOT, "profiles", "compute_profile.csv"),
        help="Path to save compute CSV profile",
    )
    args = parser.parse_args()

    local_results = profile_node(args.config, warmup_steps=args.warmup, bench_steps=args.steps)
    all_results = comm.gather(local_results, root=0)

    if rank == 0 and all_results:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        final_dict = {}
        csv_rows = []

        for node_res in sorted(all_results, key=lambda x: x["rank"]):
            nid = node_res["node_id"]
            final_dict[nid] = node_res
            for b_str, p in node_res["profiles"].items():
                csv_rows.append({
                    "node_id": nid,
                    "rank": node_res["rank"],
                    "host": node_res["host"],
                    "device_type": node_res["device_type"],
                    "batch_size": p["batch_size"],
                    "accum_steps": p["accum_steps"],
                    "mean_compute_ms": f"{p['mean_compute_ms']:.3f}",
                    "std_compute_ms": f"{p['std_compute_ms']:.3f}",
                    "throughput_img_s": f"{p['throughput_img_s']:.2f}",
                })

        # Save JSON
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(final_dict, f, indent=2)

        # Save CSV
        fieldnames = [
            "node_id", "rank", "host", "device_type", "batch_size",
            "accum_steps", "mean_compute_ms", "std_compute_ms", "throughput_img_s"
        ]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print("=" * 80)
        print(f"[SUCCESS] Compute profile saved to:\n  - {args.output_json}\n  - {args.output_csv}")
        print("=" * 80)


if __name__ == "__main__":
    main()
