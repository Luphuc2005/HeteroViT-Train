#!/usr/bin/env python3
"""Profiling sweep script for V6 Dynamic Workload Scheduler.

Runs 8 candidate CPU-GPU splits under identical hardware and training conditions:
- Candidates: 224/32, 228/28, 230/26, 232/24, 234/22, 236/20, 238/18, 240/16
- Global batch always equals 256.
- Phase 2: Precompiles all candidate GPU shapes on MirroredStrategy so function cache is hot.
- Phase 1 & 3: For each candidate split, runs 10 warm-up steps + 50 stable measurement steps.
- Phase 4: Gradient weighting strictly follows actual sample split:
    g_global = (B_gpu * g_gpu + B_cpu * g_cpu) / 256.
- Phase 5: CPU writes metrics into step_metrics_arr before cpu_done_event.set();
    no extra synchronization event.
- Outputs results/v6_profile_raw.csv and results/v6_profile_map.csv.
"""
import argparse
import csv
from multiprocessing import shared_memory
import multiprocessing as mp
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_config

CANDIDATE_SPLITS: List[Tuple[int, int]] = [
    (224, 32),
    (228, 28),
    (230, 26),
    (232, 24),
    (234, 22),
    (236, 20),
    (238, 18),
    (240, 16),
]
GLOBAL_BATCH = 256
WARMUP_STEPS_PER_SPLIT = 10
STABLE_STEPS_PER_SPLIT = 50


def _gpu_worker_sweep(
    config: Dict[str, Any],
    initial_weights_path: str,
    shapes: Tuple[Tuple[int, ...], ...],
    sizes: Tuple[int, ...],
    total_params: int,
    shm_gpu_name: str,
    shm_cpu_name: str,
    shm_global_name: str,
    sync_barrier: Any,
    step_start_barrier: Any,
    gpu_done_event: Any,
    cpu_done_event: Any,
    agg_done_event: Any,
    step_end_barrier: Any,
    step_metrics_arr: Any,
    split_info_arr: Any,
    raw_csv_path: str,
    map_csv_path: str,
):
    """GPU Worker process: Runs 2 Titan Z GPUs using MirroredStrategy."""
    joint_cfg = config.get("joint", {})
    gpu_cfg = joint_cfg.get("gpu_worker", {})
    cuda_devices = str(gpu_cfg.get("cuda_devices", "0,1"))
    cpu_cores = gpu_cfg.get("cpu_cores", [0, 1, 2, 3])
    intra_threads = int(gpu_cfg.get("intra_op_threads", 4))
    inter_threads = int(gpu_cfg.get("inter_op_threads", 2))

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["MKL_NUM_THREADS"] = str(intra_threads)

    try:
        os.sched_setaffinity(0, set(cpu_cores))
    except Exception:
        pass

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)

    from src.data.augmentation import get_train_augmentation
    from src.data.cifar10 import load_cifar10_raw
    from src.models.vit import build_vit_from_config
    from src.training.base_trainer import get_optimizer

    strategy = tf.distribute.MirroredStrategy()

    training_cfg = config.get("training", {})
    seed = int(config.get("seed", 42))

    with strategy.scope():
        model = build_vit_from_config(config)
        _ = model(tf.zeros([1, 32, 32, 3]), training=False)
        model.load_weights(initial_weights_path)
        optimizer = get_optimizer(training_cfg)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    shm_gpu = shared_memory.SharedMemory(name=shm_gpu_name)
    shm_cpu = shared_memory.SharedMemory(name=shm_cpu_name)
    shm_global = shared_memory.SharedMemory(name=shm_global_name)

    flat_gpu = np.ndarray((total_params,), dtype=np.float32, buffer=shm_gpu.buf)
    flat_cpu = np.ndarray((total_params,), dtype=np.float32, buffer=shm_cpu.buf)
    flat_global = np.ndarray((total_params,), dtype=np.float32, buffer=shm_global.buf)

    data_dir = config.get("dataset", {}).get("data_dir", "./data/cifar10")
    val_split = 0.1
    x_train_all, y_train_all, _, _ = load_cifar10_raw(data_dir)
    num_total_train = len(x_train_all)
    num_val = int(num_total_train * val_split)

    rng = np.random.default_rng(seed)
    perm_split = rng.permutation(num_total_train)
    train_idx = perm_split[num_val:]
    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx].astype(np.int32)
    num_train = len(x_train)

    train_aug = get_train_augmentation(32)

    def dynamic_step_fn(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            per_example_loss = loss_fn(labels, predictions)
            local_count = tf.cast(tf.shape(per_example_loss)[0], tf.float32)
            loss = tf.math.divide_no_nan(tf.reduce_sum(per_example_loss), local_count)
        grads = tape.gradient(loss, model.trainable_variables)
        pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
        return loss, acc, grads

    @tf.function(reduce_retracing=True)
    def dist_compute_grads_dynamic(images, labels):
        losses, accs, grads = strategy.run(dynamic_step_fn, args=(images, labels))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, losses, axis=None)
        total_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, accs, axis=None)
        reduced_grads = [
            strategy.reduce(tf.distribute.ReduceOp.MEAN, grad, axis=None)
            for grad in grads
        ]
        flat_gpu_grad = tf.concat([tf.reshape(grad, [-1]) for grad in reduced_grads], axis=0)
        return total_loss, total_acc, flat_gpu_grad

    @tf.function
    def dist_apply_grads(flat_global_grads):
        def apply_fn(flat_g):
            split_grads = tf.split(flat_g, sizes, axis=0)
            reconstructed_grads = [
                tf.reshape(g, sh) for g, sh in zip(split_grads, shapes)
            ]
            optimizer.apply_gradients(zip(reconstructed_grads, model.trainable_variables))

        strategy.run(apply_fn, args=(flat_global_grads,))

    # Phase 2: Precompile all GPU candidate shapes
    print("[GPU Worker] Precompiling all candidate GPU shapes into TensorFlow graph cache...", flush=True)
    for cand_gpu_batch, _ in CANDIDATE_SPLITS:
        t_pre_0 = time.perf_counter()
        dummy_ds = tf.data.Dataset.from_tensor_slices((
            tf.zeros([cand_gpu_batch, 32, 32, 3], dtype=tf.float32),
            tf.zeros([cand_gpu_batch], dtype=tf.int32),
        )).batch(cand_gpu_batch)
        dist_dummy = strategy.experimental_distribute_dataset(dummy_ds)
        for d_img, d_lbl in dist_dummy:
            _ = dist_compute_grads_dynamic(d_img, d_lbl)
        dt_pre = (time.perf_counter() - t_pre_0) * 1000.0
        print(f"  -> Precompiled GPU batch {cand_gpu_batch} (took {dt_pre:.1f}ms)", flush=True)

    print("[GPU Worker] Ready. Waiting for CPU worker initial sync...", flush=True)
    sync_barrier.wait()

    raw_records: List[Dict[str, Any]] = []

    # Dataset streaming pointer
    data_cursor = 0
    perm = rng.permutation(num_train)

    for split_idx, (gpu_batch, cpu_batch) in enumerate(CANDIDATE_SPLITS, start=1):
        split_id = f"{gpu_batch}/{cpu_batch}"
        w_gpu = float(gpu_batch) / float(GLOBAL_BATCH)

        # Notify CPU worker of the active split via split_info_arr: [gpu_batch, cpu_batch, split_idx]
        with split_info_arr.get_lock():
            split_info_arr[0] = gpu_batch
            split_info_arr[1] = cpu_batch
            split_info_arr[2] = split_idx

        print(f"\n[{split_idx}/{len(CANDIDATE_SPLITS)}] Running Split {split_id} "
              f"({WARMUP_STEPS_PER_SPLIT} warmup + {STABLE_STEPS_PER_SPLIT} stable steps)...", flush=True)

        total_steps_this_split = WARMUP_STEPS_PER_SPLIT + STABLE_STEPS_PER_SPLIT

        for step in range(1, total_steps_this_split + 1):
            is_warmup = (step <= WARMUP_STEPS_PER_SPLIT)
            stable_step_num = step - WARMUP_STEPS_PER_SPLIT

            # Fetch slice for GPU
            if data_cursor + GLOBAL_BATCH > num_train:
                perm = rng.permutation(num_train)
                data_cursor = 0
            step_perm = perm[data_cursor : data_cursor + GLOBAL_BATCH]
            data_cursor += GLOBAL_BATCH

            gpu_slice_idx = step_perm[:gpu_batch]
            gpu_images = x_train[gpu_slice_idx]
            gpu_labels = y_train[gpu_slice_idx]

            gpu_ds = tf.data.Dataset.from_tensor_slices((gpu_images, gpu_labels))
            gpu_ds = gpu_ds.map(train_aug, num_parallel_calls=tf.data.AUTOTUNE)
            gpu_ds = gpu_ds.batch(gpu_batch, drop_remainder=True)
            dist_gpu_ds = strategy.experimental_distribute_dataset(gpu_ds)

            # Step start sync
            gpu_done_event.clear()
            agg_done_event.clear()
            step_start_barrier.wait()

            step_wall_start = time.perf_counter()

            for images, labels in dist_gpu_ds:
                t_comp_start = time.perf_counter()
                gpu_loss, gpu_acc, flat_gpu_grad = dist_compute_grads_dynamic(images, labels)
                # Mathematical gradient weighting:
                np.multiply(flat_gpu_grad.numpy(), w_gpu, out=flat_global)
                t_comp_end = time.perf_counter()
                gpu_step_ms = (t_comp_end - t_comp_start) * 1000.0

            # Signal GPU done and wait for CPU
            gpu_done_event.set()
            t_wait_start = time.perf_counter()
            cpu_done_event.wait()
            t_wait_end = time.perf_counter()
            gpu_idle_ms = (t_wait_end - t_wait_start) * 1000.0

            # Read CPU step time and metrics from shared array (CPU wrote before setting cpu_done_event)
            cpu_step_ms = step_metrics_arr[0]
            cpu_idle_ms = step_metrics_arr[1]

            # In-Place Vector Addition
            t_merge_0 = time.perf_counter()
            np.add(flat_global, flat_cpu, out=flat_global)
            agg_done_event.set()
            grad_merge_ms = (time.perf_counter() - t_merge_0) * 1000.0

            # Apply global gradients on GPU
            t_gapply_0 = time.perf_counter()
            flat_global_tensor = tf.convert_to_tensor(flat_global, dtype=tf.float32)
            dist_apply_grads(flat_global_tensor)
            # Read 1 variable scalar to ensure GPU execution finishes
            _ = model.trainable_variables[0].read_value()[0].numpy()
            gpu_apply_ms = (time.perf_counter() - t_gapply_0) * 1000.0

            # Step boundary sync
            step_end_barrier.wait()
            cpu_apply_ms = step_metrics_arr[8]

            step_wall_ms = (time.perf_counter() - step_wall_start) * 1000.0
            throughput_img_s = GLOBAL_BATCH / max(step_wall_ms / 1000.0, 1e-6)
            critical_ms = max(gpu_step_ms, cpu_step_ms)
            delta_ms = abs(gpu_step_ms - cpu_step_ms)

            if not is_warmup:
                record = {
                    "split_id": split_id,
                    "gpu_batch": gpu_batch,
                    "cpu_batch": cpu_batch,
                    "step": stable_step_num,
                    "gpu_time_ms": gpu_step_ms,
                    "cpu_time_ms": cpu_step_ms,
                    "critical_ms": critical_ms,
                    "delta_ms": delta_ms,
                    "gpu_idle_ms": gpu_idle_ms,
                    "cpu_idle_ms": cpu_idle_ms,
                    "step_wall_ms": step_wall_ms,
                    "throughput_img_s": throughput_img_s,
                    "merge_ms": grad_merge_ms,
                    "gpu_apply_ms": gpu_apply_ms,
                    "cpu_apply_ms": cpu_apply_ms,
                }
                raw_records.append(record)

            if step == WARMUP_STEPS_PER_SPLIT:
                print(f"  Warmup done. Starting 50 stable measurements...", flush=True)
            elif not is_warmup and stable_step_num % 10 == 0:
                print(f"  [Step {stable_step_num:2d}/50] GPU: {gpu_step_ms:5.1f}ms | "
                      f"CPU: {cpu_step_ms:5.1f}ms | Crit: {critical_ms:5.1f}ms | "
                      f"Delta: {delta_ms:4.1f}ms | Wall: {step_wall_ms:5.1f}ms | "
                      f"{throughput_img_s:6.1f} img/s", flush=True)

    # Export raw data CSV
    raw_columns = [
        "split_id", "gpu_batch", "cpu_batch", "step",
        "gpu_time_ms", "cpu_time_ms", "critical_ms", "delta_ms",
        "gpu_idle_ms", "cpu_idle_ms", "step_wall_ms", "throughput_img_s",
        "merge_ms", "gpu_apply_ms", "cpu_apply_ms",
    ]
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    with open(raw_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=raw_columns)
        writer.writeheader()
        writer.writerows(raw_records)
    print(f"\n[GPU Worker] Raw profiling measurements saved to: {raw_csv_path}", flush=True)

    # Export aggregate profile map CSV
    aggregate_rows: List[Dict[str, Any]] = []
    # Find baseline 232/24 critical time for speedup calculation
    baseline_critical = 0.0
    for cand_gpu, cand_cpu in CANDIDATE_SPLITS:
        if (cand_gpu, cand_cpu) == (232, 24):
            recs = [r for r in raw_records if r["gpu_batch"] == 232 and r["cpu_batch"] == 24]
            if recs:
                baseline_critical = float(np.mean([r["critical_ms"] for r in recs]))

    for cand_gpu, cand_cpu in CANDIDATE_SPLITS:
        split_recs = [r for r in raw_records if r["gpu_batch"] == cand_gpu and r["cpu_batch"] == cand_cpu]
        if not split_recs:
            continue

        gpu_times = [r["gpu_time_ms"] for r in split_recs]
        cpu_times = [r["cpu_time_ms"] for r in split_recs]
        crit_times = [r["critical_ms"] for r in split_recs]
        deltas = [r["delta_ms"] for r in split_recs]
        walls = [r["step_wall_ms"] for r in split_recs]
        tps = [r["throughput_img_s"] for r in split_recs]
        merges = [r["merge_ms"] for r in split_recs]
        g_applies = [r["gpu_apply_ms"] for r in split_recs]
        c_applies = [r["cpu_apply_ms"] for r in split_recs]

        mean_crit = float(np.mean(crit_times))
        speedup_pct = (
            ((baseline_critical - mean_crit) / baseline_critical * 100.0)
            if baseline_critical > 0.0 else 0.0
        )

        agg_row = {
            "gpu_batch": cand_gpu,
            "cpu_batch": cand_cpu,
            "gpu_time_mean_ms": float(np.mean(gpu_times)),
            "gpu_time_std_ms": float(np.std(gpu_times)),
            "cpu_time_mean_ms": float(np.mean(cpu_times)),
            "cpu_time_std_ms": float(np.std(cpu_times)),
            "critical_mean_ms": mean_crit,
            "critical_std_ms": float(np.std(crit_times)),
            "critical_median_ms": float(np.median(crit_times)),
            "critical_p95_ms": float(np.percentile(crit_times, 95.0)),
            "delta_mean_ms": float(np.mean(deltas)),
            "delta_std_ms": float(np.std(deltas)),
            "step_wall_mean_ms": float(np.mean(walls)),
            "throughput_mean_img_s": float(np.mean(tps)),
            "throughput_std_img_s": float(np.std(tps)),
            "throughput_median_img_s": float(np.median(tps)),
            "merge_mean_ms": float(np.mean(merges)),
            "gpu_apply_mean_ms": float(np.mean(g_applies)),
            "cpu_apply_mean_ms": float(np.mean(c_applies)),
            "speedup_vs_232_24_pct": speedup_pct,
        }
        aggregate_rows.append(agg_row)

    map_columns = list(aggregate_rows[0].keys())
    with open(map_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=map_columns)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"[GPU Worker] Aggregate profile map saved to: {map_csv_path}", flush=True)

    shm_gpu.close()
    shm_cpu.close()
    shm_global.close()


def _cpu_worker_sweep(
    config: Dict[str, Any],
    initial_weights_path: str,
    shapes: Tuple[Tuple[int, ...], ...],
    sizes: Tuple[int, ...],
    total_params: int,
    shm_gpu_name: str,
    shm_cpu_name: str,
    shm_global_name: str,
    sync_barrier: Any,
    step_start_barrier: Any,
    gpu_done_event: Any,
    cpu_done_event: Any,
    agg_done_event: Any,
    step_end_barrier: Any,
    step_metrics_arr: Any,
    split_info_arr: Any,
):
    """CPU Worker process: Dedicated 18 cores for CPU training."""
    joint_cfg = config.get("joint", {})
    cpu_cfg = joint_cfg.get("cpu_worker", {})
    cpu_cores = cpu_cfg.get("cpu_cores", list(range(4, 22)))
    intra_threads = int(cpu_cfg.get("intra_op_threads", 18))
    inter_threads = int(cpu_cfg.get("inter_op_threads", 2))

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["MKL_NUM_THREADS"] = str(intra_threads)

    try:
        os.sched_setaffinity(0, set(cpu_cores))
    except Exception:
        pass

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)

    from src.data.augmentation import get_train_augmentation
    from src.data.cifar10 import load_cifar10_raw
    from src.models.vit import build_vit_from_config
    from src.training.base_trainer import get_optimizer

    training_cfg = config.get("training", {})
    seed = int(config.get("seed", 42))

    with tf.device("/CPU:0"):
        model = build_vit_from_config(config)
        _ = model(tf.zeros([1, 32, 32, 3]), training=False)
        model.load_weights(initial_weights_path)
        optimizer = get_optimizer(training_cfg)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    shm_cpu = shared_memory.SharedMemory(name=shm_cpu_name)
    shm_global = shared_memory.SharedMemory(name=shm_global_name)

    flat_cpu = np.ndarray((total_params,), dtype=np.float32, buffer=shm_cpu.buf)
    flat_global = np.ndarray((total_params,), dtype=np.float32, buffer=shm_global.buf)

    data_dir = config.get("dataset", {}).get("data_dir", "./data/cifar10")
    val_split = 0.1
    x_train_all, y_train_all, _, _ = load_cifar10_raw(data_dir)
    num_total_train = len(x_train_all)
    num_val = int(num_total_train * val_split)

    rng = np.random.default_rng(seed)
    perm_split = rng.permutation(num_total_train)
    train_idx = perm_split[num_val:]
    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx].astype(np.int32)
    num_train = len(x_train)

    train_aug = get_train_augmentation(32)

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, 32, 32, 3], tf.float32),
            tf.TensorSpec([None], tf.int32),
        ]
    )
    def cpu_compute_grads_dynamic(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
        flat_cpu_grad = tf.concat([tf.reshape(grad, [-1]) for grad in grads], axis=0)
        return loss, acc, flat_cpu_grad

    @tf.function
    def cpu_apply_grads(flat_global_grads):
        split_grads = tf.split(flat_global_grads, sizes, axis=0)
        reconstructed_grads = [
            tf.reshape(g, sh) for g, sh in zip(split_grads, shapes)
        ]
        optimizer.apply_gradients(zip(reconstructed_grads, model.trainable_variables))

    # Precompile CPU graph once with symbolic None batch
    print("[CPU Worker] Precompiling CPU dynamic graph with input_signature...", flush=True)
    t_cpu_pre = time.perf_counter()
    _ = cpu_compute_grads_dynamic(
        tf.zeros([24, 32, 32, 3], dtype=tf.float32),
        tf.zeros([24], dtype=tf.int32),
    )
    print(f"[CPU Worker] Precompiled in {(time.perf_counter() - t_cpu_pre)*1000:.1f}ms. Ready.", flush=True)

    sync_barrier.wait()

    data_cursor = 0
    perm = rng.permutation(num_train)

    for split_idx, (cand_gpu_batch, cand_cpu_batch) in enumerate(CANDIDATE_SPLITS, start=1):
        w_cpu = float(cand_cpu_batch) / float(GLOBAL_BATCH)
        total_steps_this_split = WARMUP_STEPS_PER_SPLIT + STABLE_STEPS_PER_SPLIT

        for step in range(1, total_steps_this_split + 1):
            if data_cursor + GLOBAL_BATCH > num_train:
                perm = rng.permutation(num_train)
                data_cursor = 0
            step_perm = perm[data_cursor : data_cursor + GLOBAL_BATCH]
            data_cursor += GLOBAL_BATCH

            cpu_slice_idx = step_perm[cand_gpu_batch:GLOBAL_BATCH]
            cpu_images = x_train[cpu_slice_idx]
            cpu_labels = y_train[cpu_slice_idx]

            cpu_ds = tf.data.Dataset.from_tensor_slices((cpu_images, cpu_labels))
            cpu_ds = cpu_ds.map(train_aug, num_parallel_calls=2)
            cpu_ds = cpu_ds.batch(cand_cpu_batch, drop_remainder=True)

            cpu_done_event.clear()
            step_start_barrier.wait()

            for images, labels in cpu_ds:
                t_comp_start = time.perf_counter()
                cpu_loss, cpu_acc, flat_cpu_grad = cpu_compute_grads_dynamic(images, labels)
                np.multiply(flat_cpu_grad.numpy(), w_cpu, out=flat_cpu)
                t_comp_end = time.perf_counter()
                cpu_step_ms = (t_comp_end - t_comp_start) * 1000.0

            # Phase 5: CPU writes compute metrics BEFORE signaling cpu_done_event
            step_metrics_arr[0] = cpu_step_ms
            step_metrics_arr[2] = float(cpu_loss)
            step_metrics_arr[3] = float(cpu_acc)

            # Signal CPU done and wait for GPU
            t_wait_start = time.perf_counter()
            cpu_done_event.set()
            gpu_done_event.wait()
            t_wait_end = time.perf_counter()
            cpu_idle_ms = (t_wait_end - t_wait_start) * 1000.0
            step_metrics_arr[1] = cpu_idle_ms

            # Wait for GPU worker to complete in-place aggregation
            agg_done_event.wait()

            # Apply global gradients on CPU
            t_capply_0 = time.perf_counter()
            flat_global_tensor = tf.convert_to_tensor(flat_global, dtype=tf.float32)
            cpu_apply_grads(flat_global_tensor)
            cpu_apply_ms = (time.perf_counter() - t_capply_0) * 1000.0
            step_metrics_arr[8] = cpu_apply_ms

            step_end_barrier.wait()

    shm_cpu.close()
    shm_global.close()


def run_profiling_sweep(config_path: str):
    config = load_config(config_path)

    print("============================================================================")
    print(">>> V6 Profiling Sweep (Offline Candidate Profiling)")
    print(f"Candidates ({len(CANDIDATE_SPLITS)} splits): {CANDIDATE_SPLITS}")
    print(f"Schedule: {WARMUP_STEPS_PER_SPLIT} warm-up steps + {STABLE_STEPS_PER_SPLIT} stable steps per split")
    print("============================================================================")

    # Determine parameter shapes
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    from src.models.vit import build_vit_from_config

    model_meta = build_vit_from_config(config)
    _ = model_meta(tf.zeros([1, 32, 32, 3]), training=False)

    shapes = tuple(tuple(v.shape.as_list()) for v in model_meta.trainable_variables)
    sizes = tuple(int(np.prod(s)) for s in shapes)
    total_params = sum(sizes)
    print(f"[Coordinator] Model Parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")

    with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as f:
        initial_weights_path = f.name
    model_meta.save_weights(initial_weights_path)
    del model_meta

    # Shared memory buffers
    shm_gpu = shared_memory.SharedMemory(create=True, size=total_params * 4)
    shm_cpu = shared_memory.SharedMemory(create=True, size=total_params * 4)
    shm_global = shared_memory.SharedMemory(create=True, size=total_params * 4)

    mp_ctx = mp.get_context("spawn")
    sync_barrier = mp_ctx.Barrier(2)
    step_start_barrier = mp_ctx.Barrier(2)
    gpu_done_event = mp_ctx.Event()
    cpu_done_event = mp_ctx.Event()
    agg_done_event = mp_ctx.Event()
    step_end_barrier = mp_ctx.Barrier(2)

    # [gpu_batch, cpu_batch, split_idx]
    split_info_arr = mp_ctx.Array("i", [232, 24, 0], lock=True)
    # step_metrics_arr: [cpu_step_ms, cpu_idle_ms, cpu_loss, cpu_acc, ..., cpu_apply_ms]
    step_metrics_arr = mp_ctx.Array("d", 16)

    raw_csv_path = os.path.join(PROJECT_ROOT, "results", "v6_profile_raw.csv")
    map_csv_path = os.path.join(PROJECT_ROOT, "results", "v6_profile_map.csv")

    p_gpu = mp_ctx.Process(
        target=_gpu_worker_sweep,
        args=(
            config,
            initial_weights_path,
            shapes,
            sizes,
            total_params,
            shm_gpu.name,
            shm_cpu.name,
            shm_global.name,
            sync_barrier,
            step_start_barrier,
            gpu_done_event,
            cpu_done_event,
            agg_done_event,
            step_end_barrier,
            step_metrics_arr,
            split_info_arr,
            raw_csv_path,
            map_csv_path,
        ),
    )

    p_cpu = mp_ctx.Process(
        target=_cpu_worker_sweep,
        args=(
            config,
            initial_weights_path,
            shapes,
            sizes,
            total_params,
            shm_gpu.name,
            shm_cpu.name,
            shm_global.name,
            sync_barrier,
            step_start_barrier,
            gpu_done_event,
            cpu_done_event,
            agg_done_event,
            step_end_barrier,
            step_metrics_arr,
            split_info_arr,
        ),
    )

    try:
        p_gpu.start()
        p_cpu.start()

        p_gpu.join()
        p_cpu.join()
    finally:
        for p in [p_gpu, p_cpu]:
            if p.is_alive():
                p.terminate()
                p.join()
        shm_gpu.close()
        shm_gpu.unlink()
        shm_cpu.close()
        shm_cpu.unlink()
        shm_global.close()
        shm_global.unlink()
        if os.path.exists(initial_weights_path):
            os.remove(initial_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V6 Profiling Sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/joint/joint_2gpu_b232_cpu18_b24_maxperf_20e.yaml",
        help="Path to base configuration file",
    )
    args = parser.parse_args()
    run_profiling_sweep(args.config)
