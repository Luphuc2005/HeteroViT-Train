"""Joint Heterogeneous Trainer: 2 GPU Titan Z + Dedicated CPU Worker (18 Cores).

Synchronous Joint Data-Parallel Training:
  - GPU Worker (Rank 0): 2x Titan Z GPUs (MirroredStrategy) + 4 CPU cores, batch size 256.
  - CPU Worker (Rank 1): 18 physical CPU cores, batch size 32 or 48.
  - Step-by-step synchronous weighted-average gradient aggregation according to sample count.
  - Granular profiling: GPU step time, CPU step time, GPU idle time, combined throughput.
"""
import os
import sys
import time
import csv
from typing import Dict, Any, List, Tuple
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _setup_xla():
    """Locates and configures CUDA libdevice.10.bc for XLA if available."""
    if "XLA_FLAGS" in os.environ and "--xla_gpu_cuda_data_dir" in os.environ["XLA_FLAGS"]:
        return
    import glob
    candidates = [
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
        "/usr/local/cuda-*/nvvm/libdevice/libdevice.10.bc",
        "/usr/lib/nvidia-cuda-toolkit/libdevice/libdevice.10.bc",
    ]
    for pattern in candidates:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            cuda_dir = os.path.dirname(os.path.dirname(os.path.dirname(matches[0])))
            os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_gpu_cuda_data_dir={cuda_dir}".strip()
            break


def _build_gpu_epoch_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    gpu_batch_size: int,
    cpu_batch_size: int,
    epoch: int,
    seed: int,
    image_size: int,
):
    """Partitions CIFAR-10 into disjoint GPU stream for the current epoch."""
    import tensorflow as tf
    from src.data.augmentation import get_train_augmentation

    total_batch = gpu_batch_size + cpu_batch_size
    num_samples = len(x_train)
    num_steps = num_samples // total_batch

    rng = np.random.default_rng(seed + epoch * 1000)
    perm = rng.permutation(num_samples)

    gpu_indices = np.concatenate([
        perm[s * total_batch : s * total_batch + gpu_batch_size]
        for s in range(num_steps)
    ])

    train_aug = get_train_augmentation(image_size)

    with tf.device("/CPU:0"):
        gpu_ds = tf.data.Dataset.from_tensor_slices((x_train[gpu_indices], y_train[gpu_indices]))
        gpu_ds = gpu_ds.map(train_aug, num_parallel_calls=tf.data.AUTOTUNE)
        gpu_ds = gpu_ds.batch(gpu_batch_size, drop_remainder=True)
        gpu_ds = gpu_ds.prefetch(tf.data.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        gpu_ds = gpu_ds.with_options(options)

    return gpu_ds, num_steps


def _build_cpu_epoch_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    gpu_batch_size: int,
    cpu_batch_size: int,
    epoch: int,
    seed: int,
    image_size: int,
):
    """Partitions CIFAR-10 into disjoint CPU stream for the current epoch."""
    import tensorflow as tf
    from src.data.augmentation import get_train_augmentation

    total_batch = gpu_batch_size + cpu_batch_size
    num_samples = len(x_train)
    num_steps = num_samples // total_batch

    rng = np.random.default_rng(seed + epoch * 1000)
    perm = rng.permutation(num_samples)

    cpu_indices = np.concatenate([
        perm[s * total_batch + gpu_batch_size : (s + 1) * total_batch]
        for s in range(num_steps)
    ])

    train_aug = get_train_augmentation(image_size)

    with tf.device("/CPU:0"):
        cpu_ds = tf.data.Dataset.from_tensor_slices((x_train[cpu_indices], y_train[cpu_indices]))
        cpu_ds = cpu_ds.map(train_aug, num_parallel_calls=2)
        cpu_ds = cpu_ds.batch(cpu_batch_size, drop_remainder=True)
        cpu_ds = cpu_ds.prefetch(tf.data.AUTOTUNE)

    return cpu_ds, num_steps


def _gpu_worker_proc(
    config: Dict[str, Any],
    initial_weights_path: str,
    shapes: List[List[int]],
    sizes: List[int],
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
    val_metrics_arr: Any,
    results_dir: str,
):
    """GPU Worker process: Runs 2 Titan Z GPUs using MirroredStrategy."""
    joint_cfg = config.get("joint", {})
    gpu_cfg = joint_cfg.get("gpu_worker", {})
    cuda_devices = str(gpu_cfg.get("cuda_devices", "0,1"))
    gpu_cores = gpu_cfg.get("cpu_cores", [0, 1, 2, 3])
    intra_threads = int(gpu_cfg.get("intra_op_threads", 4))
    inter_threads = int(gpu_cfg.get("inter_op_threads", 2))

    # 1. Environment and core affinity isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["MKL_NUM_THREADS"] = str(intra_threads)
    _setup_xla()
    try:
        os.sched_setaffinity(0, set(gpu_cores))
    except Exception as e:
        print(f"[GPU Worker] Warning setting affinity: {e}")

    # 2. Import TensorFlow after setting CUDA devices & affinity
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except Exception:
        pass

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    from src.models.vit import build_vit_from_config
    from src.training.base_trainer import get_optimizer
    from src.data.cifar10 import load_cifar10_raw
    from src.data.augmentation import get_val_augmentation

    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync

    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 20))
    gpu_batch_size = int(gpu_cfg.get("batch_size", 256))
    cpu_batch_size = int(joint_cfg.get("cpu_worker", {}).get("batch_size", 32))
    total_batch = gpu_batch_size + cpu_batch_size
    w_gpu = float(gpu_batch_size) / float(total_batch)
    w_cpu = float(cpu_batch_size) / float(total_batch)
    seed = int(config.get("seed", 42))

    # 3. Model, Optimizer, Loss inside Strategy Scope
    with strategy.scope():
        model = build_vit_from_config(config)
        _ = model(tf.zeros([1, 32, 32, 3]), training=False)
        model.load_weights(initial_weights_path)
        optimizer = get_optimizer(training_cfg)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE,
        )
        dummy_d2h_gpu = tf.zeros((total_params,), dtype=tf.float32)

    # 4. Attach Shared Memory
    shm_gpu = shared_memory.SharedMemory(name=shm_gpu_name)
    shm_cpu = shared_memory.SharedMemory(name=shm_cpu_name)
    shm_global = shared_memory.SharedMemory(name=shm_global_name)

    flat_gpu = np.ndarray((total_params,), dtype=np.float32, buffer=shm_gpu.buf)
    flat_cpu = np.ndarray((total_params,), dtype=np.float32, buffer=shm_cpu.buf)
    flat_global = np.ndarray((total_params,), dtype=np.float32, buffer=shm_global.buf)

    # 5. Load raw CIFAR-10 data
    data_dir = config.get("dataset", {}).get("data_dir", "./data/cifar10")
    val_split = 0.1
    x_train_all, y_train_all, x_test, y_test = load_cifar10_raw(data_dir)
    num_total_train = len(x_train_all)
    num_val = int(num_total_train * val_split)

    rng = np.random.default_rng(seed)
    perm_split = rng.permutation(num_total_train)
    val_idx, train_idx = perm_split[:num_val], perm_split[num_val:]
    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]

    # Validation dataset (evaluated on GPU)
    val_aug = get_val_augmentation(32)
    with tf.device("/CPU:0"):
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_ds = val_ds.map(val_aug, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(gpu_batch_size, drop_remainder=False)
        val_opt = tf.data.Options()
        val_opt.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        val_ds = val_ds.with_options(val_opt)
    val_ds = strategy.experimental_distribute_dataset(val_ds)

    shapes = tuple(tuple(s) for s in shapes)
    sizes = tuple(sizes)

    # Define training step functions
    def step_fn(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            per_example_loss = loss_fn(labels, predictions)
            loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=gpu_batch_size)
        grads = tape.gradient(loss, model.trainable_variables)
        pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
        return loss, acc, grads

    @tf.function
    def dist_compute_grads(images, labels):
        losses, accs, grads = strategy.run(step_fn, args=(images, labels))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
        total_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, accs, axis=None)
        reduced_grads = [
            strategy.reduce(tf.distribute.ReduceOp.SUM, g, axis=None)
            for g in grads
        ]
        flat_gpu_grad = tf.concat([tf.reshape(g, [-1]) for g in reduced_grads], axis=0)
        return total_loss, total_acc, flat_gpu_grad

    @tf.function
    def dist_apply_grads(flat_global_grads):
        def apply_fn(flat_g):
            split_grads = tf.split(flat_g, sizes, axis=0)
            reconstructed_grads = [
                tf.reshape(g, sh) for g, sh in zip(split_grads, shapes)
            ]
            try:
                optimizer.apply_gradients(
                    zip(reconstructed_grads, model.trainable_variables),
                    experimental_aggregate_gradients=False,
                )
            except TypeError:
                optimizer.apply_gradients(zip(reconstructed_grads, model.trainable_variables))
        strategy.run(apply_fn, args=(flat_global_grads,))

    def eval_val_step(images, labels):
        predictions = model(images, training=False)
        per_example_loss = loss_fn(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=gpu_batch_size)
        pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
        return loss, acc

    @tf.function
    def dist_val_step(images, labels):
        losses, accs = strategy.run(eval_val_step, args=(images, labels))
        total_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
        total_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, accs, axis=None)
        return total_loss, total_acc

    # Setup CSV files
    train_csv_path = os.path.join(results_dir, "train.csv")
    with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
            "epoch_time", "samples_per_sec", "avg_gpu_step_ms", "avg_cpu_step_ms", "avg_gpu_idle_ms"
        ])

    step_csv_path = os.path.join(results_dir, "step_metrics.csv")
    with open(step_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "step", "gpu_step_time_ms", "cpu_step_time_ms",
            "gpu_idle_time_ms", "cpu_idle_time_ms", "step_wall_time_ms",
            "combined_throughput", "train_loss", "train_acc"
        ])

    breakdown_csv_path = os.path.join(results_dir, "step_breakdown.csv")
    with open(breakdown_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "step", "step_wall_ms", "gpu_comp_ms", "gpu_d2h_ms", "gpu_idle_ms",
            "cpu_step_ms", "cpu_idle_ms", "grad_merge_ms", "h2d_ms", "gpu_apply_ms",
            "cpu_apply_ms", "end_barrier_ms", "data_iter_ms"
        ])

    print(f"[GPU Worker] Ready with 2 Titan Z GPUs ({num_replicas} replicas). Global Batch: {gpu_batch_size}")
    sync_barrier.wait()  # Initial sync with CPU worker

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        gpu_ds, num_steps = _build_gpu_epoch_dataset(
            x_train, y_train, gpu_batch_size, cpu_batch_size, epoch, seed, 32
        )
        dist_gpu_ds = strategy.experimental_distribute_dataset(gpu_ds)

        epoch_start_time = time.perf_counter()
        total_epoch_samples = 0
        total_loss_accum = 0.0
        total_acc_accum = 0.0
        gpu_step_times = []
        cpu_step_times = []
        gpu_idle_times = []

        bd_step_wall = []
        bd_gpu_comp = []
        bd_gpu_d2h = []
        bd_gpu_idle = []
        bd_cpu_step = []
        bd_cpu_idle = []
        bd_merge = []
        bd_h2d = []
        bd_gpu_apply = []
        bd_cpu_apply = []
        bd_end_barrier = []
        bd_data_iter = []

        t_data_prev = time.perf_counter()

        for step, (images, labels) in enumerate(dist_gpu_ds, start=1):
            t_data_iter_ms = (time.perf_counter() - t_data_prev) * 1000.0

            # Synchronize start of step across both workers
            gpu_done_event.clear()
            agg_done_event.clear()
            step_start_barrier.wait()

            step_wall_start = time.perf_counter()

            # A. Compute local GPU gradients (Fused GPU tensor, single D2H copy)
            t_comp_start = time.perf_counter()
            gpu_loss, gpu_acc, flat_gpu_grad = dist_compute_grads(images, labels)
            flat_gpu[:] = flat_gpu_grad.numpy()
            t_comp_end = time.perf_counter()
            gpu_step_ms = (t_comp_end - t_comp_start) * 1000.0

            # Measure pure D2H DMA transfer speed of 10.9MB using pre-allocated dummy
            t_dma_0 = time.perf_counter()
            _ = dummy_d2h_gpu.numpy()
            gpu_d2h_ms = (time.perf_counter() - t_dma_0) * 1000.0
            gpu_pure_comp_ms = max(gpu_step_ms - gpu_d2h_ms, 0.0)

            # B. Signal GPU done and wait for CPU
            gpu_done_event.set()
            t_wait_start = time.perf_counter()
            cpu_done_event.wait()
            t_wait_end = time.perf_counter()
            gpu_idle_ms = (t_wait_end - t_wait_start) * 1000.0

            # C. Read CPU step time from shared array
            cpu_step_ms = step_metrics_arr[0]
            cpu_idle_ms = step_metrics_arr[1]
            cpu_loss = step_metrics_arr[2]
            cpu_acc = step_metrics_arr[3]

            # D. Weighted aggregation of gradients
            t_merge_0 = time.perf_counter()
            flat_global[:] = w_gpu * flat_gpu[:] + w_cpu * flat_cpu[:]
            agg_done_event.set()
            grad_merge_ms = (time.perf_counter() - t_merge_0) * 1000.0

            # E. Apply global gradients (Single 10.9MB H2D copy, GPU-side split & reshape)
            t_h2d_0 = time.perf_counter()
            flat_global_tensor = tf.convert_to_tensor(flat_global, dtype=tf.float32)
            h2d_ms = (time.perf_counter() - t_h2d_0) * 1000.0

            t_gapply_0 = time.perf_counter()
            dist_apply_grads(flat_global_tensor)
            # Sync GPU stream by reading 1 scalar to capture true GPU apply time
            _ = model.trainable_variables[0].read_value()[0].numpy()
            gpu_apply_ms = (time.perf_counter() - t_gapply_0) * 1000.0

            # Step boundary sync
            t_bend_0 = time.perf_counter()
            step_end_barrier.wait()
            end_barrier_ms = (time.perf_counter() - t_bend_0) * 1000.0

            # Read CPU apply time recorded before barrier
            cpu_apply_ms = step_metrics_arr[8]

            # Step metrics
            step_wall_ms = (time.perf_counter() - step_wall_start) * 1000.0
            step_throughput = total_batch / max(step_wall_ms / 1000.0, 1e-6)
            joint_loss = w_gpu * float(gpu_loss) + w_cpu * cpu_loss
            joint_acc = w_gpu * float(gpu_acc) + w_cpu * cpu_acc

            total_epoch_samples += total_batch
            total_loss_accum += joint_loss
            total_acc_accum += joint_acc
            gpu_step_times.append(gpu_step_ms)
            cpu_step_times.append(cpu_step_ms)
            gpu_idle_times.append(gpu_idle_ms)

            # Record breakdown metrics
            bd_step_wall.append(step_wall_ms)
            bd_gpu_comp.append(gpu_pure_comp_ms)
            bd_gpu_d2h.append(gpu_d2h_ms)
            bd_gpu_idle.append(gpu_idle_ms)
            bd_cpu_step.append(cpu_step_ms)
            bd_cpu_idle.append(cpu_idle_ms)
            bd_merge.append(grad_merge_ms)
            bd_h2d.append(h2d_ms)
            bd_gpu_apply.append(gpu_apply_ms)
            bd_cpu_apply.append(cpu_apply_ms)
            bd_end_barrier.append(end_barrier_ms)
            bd_data_iter.append(t_data_iter_ms)

            # Save step metrics
            with open(step_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, step, f"{gpu_step_ms:.1f}", f"{cpu_step_ms:.1f}",
                    f"{gpu_idle_ms:.1f}", f"{cpu_idle_ms:.1f}", f"{step_wall_ms:.1f}",
                    f"{step_throughput:.1f}", f"{joint_loss:.4f}", f"{joint_acc * 100:.2f}"
                ])

            with open(breakdown_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, step, f"{step_wall_ms:.2f}", f"{gpu_pure_comp_ms:.2f}",
                    f"{gpu_d2h_ms:.2f}", f"{gpu_idle_ms:.2f}", f"{cpu_step_ms:.2f}",
                    f"{cpu_idle_ms:.2f}", f"{grad_merge_ms:.2f}", f"{h2d_ms:.2f}",
                    f"{gpu_apply_ms:.2f}", f"{cpu_apply_ms:.2f}", f"{end_barrier_ms:.2f}",
                    f"{t_data_iter_ms:.2f}"
                ])

            if step % 25 == 0 or step == num_steps:
                print(
                    f"  [E{epoch:02d} S{step:03d}/{num_steps:03d}] "
                    f"GPU: {gpu_step_ms:5.1f}ms | CPU: {cpu_step_ms:5.1f}ms | "
                    f"Wait: {gpu_idle_ms:4.1f}ms | {step_throughput:6.1f} img/s | "
                    f"Loss: {joint_loss:6.4f} | Acc: {joint_acc * 100:5.2f}%",
                    flush=True,
                )

            t_data_prev = time.perf_counter()

        epoch_time = time.perf_counter() - epoch_start_time
        combined_throughput = total_epoch_samples / max(epoch_time, 1e-6)
        avg_train_loss = total_loss_accum / max(num_steps, 1)
        avg_train_acc = total_acc_accum / max(num_steps, 1)
        avg_gpu_step = np.mean(gpu_step_times)
        avg_cpu_step = np.mean(cpu_step_times)
        avg_gpu_idle = np.mean(gpu_idle_times)

        # Compute decomposition averages for epoch
        m_wall = float(np.mean(bd_step_wall))
        m_gcomp = float(np.mean(bd_gpu_comp))
        m_gd2h = float(np.mean(bd_gpu_d2h))
        m_gidle = float(np.mean(bd_gpu_idle))
        m_cstep = float(np.mean(bd_cpu_step))
        m_cidle = float(np.mean(bd_cpu_idle))
        m_merge = float(np.mean(bd_merge))
        m_h2d = float(np.mean(bd_h2d))
        m_gapply = float(np.mean(bd_gpu_apply))
        m_capply = float(np.mean(bd_cpu_apply))
        m_endb = float(np.mean(bd_end_barrier))
        m_data = float(np.mean(bd_data_iter))
        m_noncomp = m_wall - (m_gcomp + m_gd2h + m_gidle)

        print(
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  >>> STEP WALL TIME DECOMPOSITION (Epoch {epoch:02d} Mean):\n"
            f"      Total Step Wall Time:    {m_wall:5.1f} ms (100.0%)\n"
            f"      ────────────────────────────────────────────────────────────────────────────\n"
            f"      [1. Compute Phase]:\n"
            f"        • GPU Compute (Fwd+Bwd+Concat): {m_gcomp:5.1f} ms ({m_gcomp/m_wall*100:4.1f}%)\n"
            f"        • GPU D2H Transfer (10.9MB):    {m_gd2h:5.1f} ms ({m_gd2h/m_wall*100:4.1f}%)\n"
            f"        • GPU Idle (Waiting CPU):       {m_gidle:5.1f} ms ({m_gidle/m_wall*100:4.1f}%)\n"
            f"        • CPU Compute Step (Ref):       {m_cstep:5.1f} ms ({m_cstep/m_wall*100:4.1f}%)\n"
            f"      [2. Non-Compute Phase (~45 ms)]:  {m_noncomp:5.1f} ms ({m_noncomp/m_wall*100:4.1f}%)\n"
            f"        • RAM Gradient Merge:           {m_merge:5.1f} ms ({m_merge/m_wall*100:4.1f}%)\n"
            f"        • H2D Tensor Conversion:        {m_h2d:5.1f} ms ({m_h2d/m_wall*100:4.1f}%)\n"
            f"        • GPU Apply (Split/AdamW/Sync): {m_gapply:5.1f} ms ({m_gapply/m_wall*100:4.1f}%)\n"
            f"        • CPU Apply (18-Core AdamW):    {m_capply:5.1f} ms ({m_capply/m_wall*100:4.1f}%)\n"
            f"        • End Barrier Wait:             {m_endb:5.1f} ms ({m_endb/m_wall*100:4.1f}%)\n"
            f"      [3. Pipeline Overhead]:\n"
            f"        • Inter-Step Data Iterator:     {m_data:5.1f} ms\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            flush=True,
        )

        # Validation phase on GPU
        val_loss_accum = 0.0
        val_acc_accum = 0.0
        val_batches = 0
        for v_imgs, v_lbls in val_ds:
            vl, va = dist_val_step(v_imgs, v_lbls)
            val_loss_accum += float(vl)
            val_acc_accum += float(va)
            val_batches += 1
        avg_val_loss = val_loss_accum / max(val_batches, 1)
        avg_val_acc = val_acc_accum / max(val_batches, 1)

        # Write to shared array for logger
        val_metrics_arr[0] = avg_train_loss
        val_metrics_arr[1] = avg_train_acc
        val_metrics_arr[2] = avg_val_loss
        val_metrics_arr[3] = avg_val_acc
        val_metrics_arr[4] = epoch_time
        val_metrics_arr[5] = combined_throughput
        val_metrics_arr[6] = avg_gpu_step
        val_metrics_arr[7] = avg_cpu_step
        val_metrics_arr[8] = avg_gpu_idle

        is_best = avg_val_acc > best_val_acc
        best_marker = " [★ BEST]" if is_best else ""
        print(
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"  >>> EPOCH {epoch:02d}/{epochs:02d} DONE ({epoch_time:4.1f}s) | "
            f"Throughput: {combined_throughput:6.1f} img/s | "
            f"GPU: {avg_gpu_step:5.1f}ms | CPU: {avg_cpu_step:5.1f}ms | Idle: {avg_gpu_idle:4.1f}ms\n"
            f"      Train Acc: {avg_train_acc * 100:5.2f}% | Val Acc: {avg_val_acc * 100:5.2f}%{best_marker}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            flush=True,
        )

        # Write epoch metrics to train.csv
        with open(train_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, f"{avg_train_loss:.5f}", f"{avg_train_acc:.5f}",
                f"{avg_val_loss:.5f}", f"{avg_val_acc:.5f}", f"{epoch_time:.2f}",
                f"{combined_throughput:.2f}", f"{avg_gpu_step:.1f}",
                f"{avg_cpu_step:.1f}", f"{avg_gpu_idle:.1f}"
            ])

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            model.save_weights(os.path.join(results_dir, "best.weights.h5"))

    # Cleanup SHM
    shm_gpu.close()
    shm_cpu.close()
    shm_global.close()


def _cpu_worker_proc(
    config: Dict[str, Any],
    initial_weights_path: str,
    shapes: List[List[int]],
    sizes: List[int],
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
):
    """CPU Worker process: Dedicated 18 cores for CPU training."""
    joint_cfg = config.get("joint", {})
    cpu_cfg = joint_cfg.get("cpu_worker", {})
    cpu_cores = cpu_cfg.get("cpu_cores", list(range(4, 22)))
    intra_threads = int(cpu_cfg.get("intra_op_threads", 18))
    inter_threads = int(cpu_cfg.get("inter_op_threads", 2))

    # 1. Environment & Core affinity isolation
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = str(intra_threads)
    os.environ["MKL_NUM_THREADS"] = str(intra_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(intra_threads)
    try:
        os.sched_setaffinity(0, set(cpu_cores))
    except Exception as e:
        print(f"[CPU Worker] Warning setting affinity: {e}")

    # 2. Import TensorFlow
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except Exception:
        pass

    from src.models.vit import build_vit_from_config
    from src.training.base_trainer import get_optimizer
    from src.data.cifar10 import load_cifar10_raw

    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 20))
    gpu_batch_size = int(joint_cfg.get("gpu_worker", {}).get("batch_size", 256))
    cpu_batch_size = int(cpu_cfg.get("batch_size", 32))
    seed = int(config.get("seed", 42))

    with tf.device("/CPU:0"):
        model = build_vit_from_config(config)
        _ = model(tf.zeros([1, 32, 32, 3]), training=False)
        model.load_weights(initial_weights_path)
        optimizer = get_optimizer(training_cfg)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Attach Shared Memory
    shm_gpu = shared_memory.SharedMemory(name=shm_gpu_name)
    shm_cpu = shared_memory.SharedMemory(name=shm_cpu_name)
    shm_global = shared_memory.SharedMemory(name=shm_global_name)

    flat_cpu = np.ndarray((total_params,), dtype=np.float32, buffer=shm_cpu.buf)
    flat_global = np.ndarray((total_params,), dtype=np.float32, buffer=shm_global.buf)

    # Load raw CIFAR-10 data
    data_dir = config.get("dataset", {}).get("data_dir", "./data/cifar10")
    val_split = 0.1
    x_train_all, y_train_all, _, _ = load_cifar10_raw(data_dir)
    num_total_train = len(x_train_all)
    num_val = int(num_total_train * val_split)

    rng = np.random.default_rng(seed)
    perm_split = rng.permutation(num_total_train)
    train_idx = perm_split[num_val:]
    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]

    shapes = tuple(tuple(s) for s in shapes)
    sizes = tuple(sizes)

    @tf.function
    def cpu_compute_grads(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_fn(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
        flat_cpu_grad = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return loss, acc, flat_cpu_grad

    @tf.function
    def cpu_apply_grads(flat_global_grads):
        split_grads = tf.split(flat_global_grads, sizes, axis=0)
        reconstructed_grads = [
            tf.reshape(g, sh) for g, sh in zip(split_grads, shapes)
        ]
        optimizer.apply_gradients(zip(reconstructed_grads, model.trainable_variables))

    print(f"[CPU Worker] Ready with {len(cpu_cores)} physical cores. Batch Size: {cpu_batch_size}")
    sync_barrier.wait()  # Initial sync with GPU worker

    for epoch in range(1, epochs + 1):
        cpu_ds, _ = _build_cpu_epoch_dataset(
            x_train, y_train, gpu_batch_size, cpu_batch_size, epoch, seed, 32
        )

        for step, (images, labels) in enumerate(cpu_ds, start=1):
            # Synchronize start of step across both workers
            cpu_done_event.clear()
            step_start_barrier.wait()

            # A. Compute local CPU gradients (Fused, single copy)
            t_comp_start = time.perf_counter()
            cpu_loss, cpu_acc, flat_cpu_grad = cpu_compute_grads(images, labels)
            flat_cpu[:] = flat_cpu_grad.numpy()
            t_comp_end = time.perf_counter()
            cpu_step_ms = (t_comp_end - t_comp_start) * 1000.0

            # B. Signal CPU done and wait for GPU
            t_wait_start = time.perf_counter()
            cpu_done_event.set()
            gpu_done_event.wait()
            t_wait_end = time.perf_counter()
            cpu_idle_ms = (t_wait_end - t_wait_start) * 1000.0

            # Store metrics in shared array for GPU worker logging
            step_metrics_arr[0] = cpu_step_ms
            step_metrics_arr[1] = cpu_idle_ms
            step_metrics_arr[2] = float(cpu_loss)
            step_metrics_arr[3] = float(cpu_acc)

            # C. Wait for GPU worker to complete weighted aggregation
            agg_done_event.wait()

            # D. Apply global gradients (Single tensor, graph-side split)
            t_capply_0 = time.perf_counter()
            flat_global_tensor = tf.convert_to_tensor(flat_global, dtype=tf.float32)
            cpu_apply_grads(flat_global_tensor)
            cpu_apply_ms = (time.perf_counter() - t_capply_0) * 1000.0
            step_metrics_arr[8] = cpu_apply_ms

            # E. Step boundary sync
            t_cbend_0 = time.perf_counter()
            step_end_barrier.wait()
            cpu_end_barrier_ms = (time.perf_counter() - t_cbend_0) * 1000.0
            step_metrics_arr[9] = cpu_end_barrier_ms

    # Cleanup SHM
    shm_gpu.close()
    shm_cpu.close()
    shm_global.close()


class JointTrainer:
    """Coordinator that sets up shared memory, initial weights, and launches GPU and CPU workers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exp_name = config.get("experiment", {}).get("name", "joint_training")
        output_dir = config.get("logging", {}).get("output_dir", "./results/joint_benchmarks")

        from datetime import datetime, timezone, timedelta
        VN_TZ = timezone(timedelta(hours=7))
        timestamp = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"{self.exp_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def train(self):
        import tensorflow as tf
        from src.models.vit import build_vit_from_config
        from src.metrics.logger import ExperimentLogger

        # Save config copy and metadata
        logger = ExperimentLogger(self.config)
        self.run_dir = logger.run_dir

        print(f"================================================================")
        print(f"=== Starting Joint 2-GPU + 18-Core CPU Training ===")
        print(f"Run directory: {self.run_dir}")
        print(f"================================================================")

        # 1. Instantiate reference model to determine parameter shapes and create initial weights
        ref_model = build_vit_from_config(self.config)
        _ = ref_model(tf.zeros([1, 32, 32, 3]), training=False)
        shapes = [v.shape.as_list() for v in ref_model.trainable_variables]
        sizes = [int(np.prod(s)) for s in shapes]
        total_params = sum(sizes)

        initial_weights_path = os.path.join(self.run_dir, "initial_weights.h5")
        ref_model.save_weights(initial_weights_path)
        del ref_model

        print(f"Model Parameters: {total_params:,} floats ({total_params * 4 / (1024*1024):.2f} MB)")

        # 2. Allocate POSIX Shared Memory for Gradients
        shm_gpu = shared_memory.SharedMemory(create=True, size=total_params * 4)
        shm_cpu = shared_memory.SharedMemory(create=True, size=total_params * 4)
        shm_global = shared_memory.SharedMemory(create=True, size=total_params * 4)

        # 3. Multiprocessing synchronization primitives
        mp_ctx = mp.get_context("spawn")
        sync_barrier = mp_ctx.Barrier(2)
        step_start_barrier = mp_ctx.Barrier(2)
        gpu_done_event = mp_ctx.Event()
        cpu_done_event = mp_ctx.Event()
        agg_done_event = mp_ctx.Event()
        step_end_barrier = mp_ctx.Barrier(2)

        # Shared arrays for metrics:
        # step_metrics_arr: [cpu_step_ms, cpu_idle_ms, cpu_loss, cpu_acc, ..., cpu_apply_ms, cpu_end_barrier_ms]
        step_metrics_arr = mp_ctx.Array("d", 16)
        # val_metrics_arr: [train_loss, train_acc, val_loss, val_acc, epoch_time, throughput, avg_gpu_step, avg_cpu_step, avg_gpu_idle]
        val_metrics_arr = mp_ctx.Array("d", 9)

        # 4. Spawn Worker Processes
        p_gpu = mp_ctx.Process(
            target=_gpu_worker_proc,
            args=(
                self.config,
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
                val_metrics_arr,
                self.run_dir,
            ),
        )

        p_cpu = mp_ctx.Process(
            target=_cpu_worker_proc,
            args=(
                self.config,
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
            ),
        )

        try:
            p_gpu.start()
            p_cpu.start()

            p_gpu.join()
            p_cpu.join()
        finally:
            # 5. Cleanup Shared Memory
            shm_gpu.close()
            shm_gpu.unlink()
            shm_cpu.close()
            shm_cpu.unlink()
            shm_global.close()
            shm_global.unlink()

        print(f"\n[Joint Training Finished] Results saved to: {self.run_dir}")
