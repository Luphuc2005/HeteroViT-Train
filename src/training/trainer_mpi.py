"""Synchronous MPI Distributed Trainer for Heterogeneous 5-Node Cluster.

Supports:
  - Rank 0: 2x NVIDIA Titan Z GPUs managed internally via tf.distribute.MirroredStrategy
  - Ranks 1-4: Multi-core CPU workers (single replica per rank)
  - Synchronous AllReduce of mean gradients across all 5 ranks
  - Complete model weight verification (model.get_weights()) and gradient verification
"""
import os
import time
import csv
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import tensorflow as tf
from mpi4py import MPI

from src.training.base_trainer import BaseTrainer, get_optimizer
from src.metrics.logger import ExperimentLogger


class MPITrainer(BaseTrainer):
    """Synchronous MPI Trainer for 5-Node Cluster with multi-GPU rank 0."""

    def __init__(
        self,
        comm: MPI.Comm,
        rank: int,
        world_size: int,
        strategy: Optional[tf.distribute.Strategy],
        model: tf.keras.Model,
        config: Dict[str, Any],
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        steps_per_epoch: int,
        val_steps: int,
        logger: ExperimentLogger,
        max_steps: Optional[int] = None,
        start_epoch: int = 1,
        best_val_accuracy: float = 0.0,
    ):
        self.comm = comm
        self.rank = rank
        self.world_size = world_size
        self.strategy = strategy
        self.max_steps = max_steps
        self.start_epoch = int(start_epoch)

        super().__init__(
            model=model,
            config=config,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            steps_per_epoch=steps_per_epoch,
            val_steps=val_steps,
            logger=logger,
        )

        self.best_val_accuracy = float(best_val_accuracy)
        training_cfg = config.get("training", {})
        rank_batch_sizes = training_cfg.get("rank_batch_sizes", None)
        if rank_batch_sizes is not None:
            self.rank_batch_sizes = [int(b) for b in rank_batch_sizes]
            self.local_batch_size = self.rank_batch_sizes[self.rank]
            self.global_batch_size = sum(self.rank_batch_sizes)
        else:
            self.local_batch_size = int(training_cfg.get("batch_size", 128))
            self.global_batch_size = self.local_batch_size * self.world_size
            self.rank_batch_sizes = [self.local_batch_size] * self.world_size

        self.sync_interval = int(training_cfg.get("sync_interval", 20))
        self.sync_tolerance = float(training_cfg.get("sync_tolerance", 1e-5))

        # Gradient accumulation configuration (e.g. for GPU rank with large batch)
        grad_accum_cfg = training_cfg.get("grad_accum", {})
        self.accum_steps = 1
        self.micro_batch_size = self.local_batch_size
        if self.rank == 0 and grad_accum_cfg.get("enabled", False):
            mb = int(grad_accum_cfg.get("micro_batch_size", 128))
            num_reps = self.strategy.num_replicas_in_sync if self.strategy is not None else 1
            rep_b = self.local_batch_size // max(1, num_reps)
            if rep_b > mb:
                divs = [d for d in range(2, rep_b + 1) if rep_b % d == 0 and rep_b // d <= mb]
                self.accum_steps = min(divs) if divs else max(1, rep_b // mb)
                self.micro_batch_size = self.local_batch_size // self.accum_steps

        import socket
        self.hostname = socket.gethostname()
        self.rank_to_node = {0: "lab01", 1: "lab02", 2: "lab03", 3: "lab04", 4: "lab05"}
        self.node_id = self.rank_to_node.get(self.rank, f"lab0{self.rank+1}")
        self.num_gpus = len(tf.config.list_physical_devices("GPU")) if self.rank == 0 else 0
        self.device_str = f"{self.num_gpus}xGPU" if (self.rank == 0 and self.num_gpus > 0) else "CPU"

        # Phase 2: Dynamic Load Balancer configuration
        self.dynamic_cfg = config.get("dynamic_scheduler", {})
        self.dynamic_rebalance_enabled = bool(self.dynamic_cfg.get("enabled", False))
        self.dynamic_rebalancer = None

        if self.rank == 0 and self.dynamic_rebalance_enabled:
            from src.scheduler.models.compute_model import ComputeCostModel
            from src.scheduler.models.communication_model import CommunicationCostModel
            from src.scheduler.online_state import OnlineClusterState
            from src.scheduler.online_cost_model import OnlineCostModel
            from src.scheduler.dynamic_rebalancer import DynamicRebalancer

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            comp_prof = self.dynamic_cfg.get("compute_profile", os.path.join(project_root, "profiles", "compute_profile.json"))
            net_prof = self.dynamic_cfg.get("network_profile", os.path.join(project_root, "profiles", "network_profile.json"))
            cluster_state = OnlineClusterState(ema_alpha=float(self.dynamic_cfg.get("ema_alpha", 0.2)))
            online_cm = OnlineCostModel(
                compute_model=ComputeCostModel(comp_prof),
                communication_model=CommunicationCostModel(net_prof),
                cluster_state=cluster_state,
                r_min=float(self.dynamic_cfg.get("r_min", 0.5)),
                r_max=float(self.dynamic_cfg.get("r_max", 3.0)),
                lambda_penalty=float(self.dynamic_cfg.get("lambda_penalty", 1.0)),
            )
            rebalance_csv = os.path.join(self.logger.run_dir, "rebalance_log.csv") if hasattr(self.logger, "run_dir") else None
            self.dynamic_rebalancer = DynamicRebalancer(
                cost_model=online_cm,
                epsilon=float(self.dynamic_cfg.get("epsilon", 0.05)),
                cooldown_epochs=int(self.dynamic_cfg.get("cooldown_epochs", 1)),
                slowdown_threshold_r=float(self.dynamic_cfg.get("slowdown_threshold_r", 1.15)),
                log_csv_path=rebalance_csv,
            )

        # Phase 1: Local-Step Synchronization configuration
        self.sync_mode = str(training_cfg.get("sync_mode", "gradient_allreduce")).lower()
        ls_cfg = training_cfg.get("local_steps", 10)
        self.local_steps = "full_epoch" if str(ls_cfg).lower() == "full_epoch" else int(ls_cfg)
        self.opt_state_sync = str(training_cfg.get("optimizer_state_sync", "preserve_local")).lower()
        self.total_train_time = 0.0

        self.timeline_csv = None
        if self.rank == 0 and hasattr(self.logger, "run_dir"):
            self.timeline_csv = os.path.join(self.logger.run_dir, "ranks_timeline.csv")
            if not os.path.exists(self.timeline_csv):
                with open(self.timeline_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch",
                        "rank",
                        "host",
                        "device",
                        "batch_size",
                        "local_loss",
                        "local_acc",
                        "compute_time_ms",
                        "idle_wait_ms",
                        "idle_pct",
                        "is_straggler",
                    ])
        # Phase 1 & 2: Step-level Runtime Instrumentation
        self.runtime_profiler = None
        if hasattr(self.logger, "run_dir") and self.logger.run_dir:
            from src.scheduler.profiling.runtime_profiler import RuntimeProfiler
            fname = "step_metrics.csv" if self.rank == 0 else f"step_metrics_rank{self.rank}.csv"
            ema_a = float(self.dynamic_cfg.get("ema_alpha", 0.2)) if self.dynamic_rebalance_enabled else 0.15
            self.runtime_profiler = RuntimeProfiler(
                output_dir=self.logger.run_dir,
                node_id=self.node_id,
                rank=self.rank,
                ema_alpha=ema_a,
                filename=fname,
            )

        # Step-level Detailed Timing Breakdown CSV (all ranks per step)
        self.step_timing_csv = None
        if self.rank == 0 and hasattr(self.logger, "run_dir") and self.logger.run_dir:
            self.step_timing_csv = os.path.join(self.logger.run_dir, "step_timing_breakdown.csv")
            if not os.path.exists(self.step_timing_csv):
                with open(self.step_timing_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch",
                        "step",
                        "rank",
                        "batch",
                        "data_ms",
                        "compute_ms",
                        "allreduce_ms",
                        "optimizer_ms",
                        "barrier_ms",
                        "other_ms",
                        "step_ms",
                    ])

        self.rounds_csv = None
        if self.rank == 0 and hasattr(self.logger, "run_dir") and self.sync_mode == "local_steps":
            self.rounds_csv = os.path.join(self.logger.run_dir, "rounds_timeline.csv")
            if not os.path.exists(self.rounds_csv):
                with open(self.rounds_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "epoch",
                        "round",
                        "rank",
                        "local_steps",
                        "batch_size",
                        "samples_processed",
                        "compute_time",
                        "wait_time",
                        "sync_time",
                        "round_time",
                        "throughput",
                        "train_loss",
                    ])

        # Re-initialize optimizer and loss inside strategy scope if rank 0 has MirroredStrategy
        if self.rank == 0 and self.strategy is not None:
            with self.strategy.scope():
                self.optimizer = get_optimizer(training_cfg)
                # Correction 1: per-example loss (reduction=NONE) before compute_average_loss
                self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE,
                )
            self.dist_train_ds = self.strategy.experimental_distribute_dataset(self.train_ds)
            self.dist_val_ds = self.strategy.experimental_distribute_dataset(self.val_ds)
        else:
            self.optimizer = get_optimizer(training_cfg)
            # Standard reduction for CPU workers: mean over local batch
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.AUTO,
            )
            self.dist_train_ds = self.train_ds
            self.dist_val_ds = self.val_ds

        self._setup_step_functions()

    def _setup_step_functions(self):
        """Builds graph-compiled local gradient computation and weight application functions."""
        local_batch = self.local_batch_size
        accum_steps = self.accum_steps

        if self.rank == 0 and self.strategy is not None:
            strategy = self.strategy
            model = self.model
            loss_fn = self.loss_fn
            optimizer = self.optimizer

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
                            # Normalize by rank 0's total local batch size
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
                def compute_local_grads(images, labels):
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

            else:
                def replica_step_fn(images, labels):
                    with tf.GradientTape() as tape:
                        predictions = model(images, training=True)
                        per_example_loss = loss_fn(labels, predictions)
                        # Normalize by rank 0's total local batch size
                        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=local_batch)
                    grads = tape.gradient(loss, model.trainable_variables)
                    pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                    acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                    return loss, grads, acc

                @tf.function
                def compute_local_grads(images, labels):
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

            @tf.function
            def apply_global_grads(global_grads):
                # Correction 2: Ensure MirroredStrategy/Keras does NOT aggregate already-averaged gradients
                def apply_fn(*grads):
                    try:
                        optimizer.apply_gradients(
                            zip(grads, model.trainable_variables),
                            experimental_aggregate_gradients=False,
                        )
                    except TypeError:
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))

                strategy.run(apply_fn, args=tuple(global_grads))

            # Phase 1: Local training step for local_steps mode (forward + backward + optimizer update)
            def replica_local_train_step(images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    per_example_loss = loss_fn(labels, predictions)
                    loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=local_batch)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                return loss, acc

            @tf.function
            def local_train_step(images, labels):
                per_replica_losses, per_replica_accs = strategy.run(
                    replica_local_train_step, args=(images, labels)
                )
                local_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                local_acc = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
                return local_loss, local_acc

            self._compute_local_grads = compute_local_grads
            self._apply_global_grads = apply_global_grads
            self._local_train_step = local_train_step

        else:
            model = self.model
            loss_fn = self.loss_fn
            optimizer = self.optimizer

            if accum_steps > 1:
                @tf.function
                def compute_local_grads(images, labels):
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
            else:
                @tf.function
                def compute_local_grads(images, labels):
                    with tf.GradientTape() as tape:
                        predictions = model(images, training=True)
                        loss = loss_fn(labels, predictions) # mean loss over local batch
                    grads = tape.gradient(loss, model.trainable_variables)
                    pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                    acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                    return loss, acc, grads

            @tf.function
            def apply_global_grads(global_grads):
                optimizer.apply_gradients(zip(global_grads, model.trainable_variables))

            @tf.function
            def local_train_step(images, labels):
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    loss = loss_fn(labels, predictions)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                acc = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                return loss, acc

            self._compute_local_grads = compute_local_grads
            self._apply_global_grads = apply_global_grads
            self._local_train_step = local_train_step

    def rebuild_dataset(self, new_rank_batch_sizes: List[int]):
        """Re-partitions and rebuilds local training dataset when allocation changes."""
        self.rank_batch_sizes = [int(b) for b in new_rank_batch_sizes]
        self.local_batch_size = self.rank_batch_sizes[self.rank]
        self.global_batch_size = sum(self.rank_batch_sizes)

        # Gradient accumulation configuration
        grad_accum_cfg = self.config.get("training", {}).get("grad_accum", {})
        self.accum_steps = 1
        self.micro_batch_size = self.local_batch_size
        if self.rank == 0 and grad_accum_cfg.get("enabled", False):
            mb = int(grad_accum_cfg.get("micro_batch_size", 128))
            num_reps = self.strategy.num_replicas_in_sync if self.strategy is not None else 1
            rep_b = self.local_batch_size // max(1, num_reps)
            if rep_b > mb:
                divs = [d for d in range(2, rep_b + 1) if rep_b % d == 0 and rep_b // d <= mb]
                self.accum_steps = min(divs) if divs else max(1, rep_b // mb)
                self.micro_batch_size = self.local_batch_size // self.accum_steps

        from src.data.cifar10 import build_cifar10_datasets
        dataset_cfg = self.config.get("dataset", {})
        data_dir = dataset_cfg.get("data_dir", "./data/cifar10")
        model_cfg = self.config.get("model", {})
        image_size = int(model_cfg.get("image_size", 32))
        seed = int(self.config.get("seed", 42))
        drop_remainder = bool(self.config.get("training", {}).get("drop_remainder", True))

        if self.local_batch_size > 0:
            train_ds, val_ds, test_ds, steps_per_epoch, val_steps = build_cifar10_datasets(
                data_dir=data_dir,
                batch_size=self.local_batch_size,
                val_split=0.1,
                image_size=image_size,
                seed=seed,
                cache=True,
                rank=self.rank,
                num_ranks=self.world_size,
                drop_remainder=drop_remainder,
                rank_batch_sizes=self.rank_batch_sizes,
            )
            self.train_ds = train_ds
            if self.rank == 0 and self.strategy is not None:
                self.dist_train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            else:
                self.dist_train_ds = train_ds
            self.steps_per_epoch = steps_per_epoch
        else:
            self.train_ds = None
            self.dist_train_ds = None
            self.steps_per_epoch = 45000 // max(self.global_batch_size, 1)

        self._setup_step_functions()

    def allreduce_gradients(
        self,
        local_grads: List[tf.Tensor],
        pre_packed_parts: Optional[List[np.ndarray]] = None,
        pre_grad_metadata: Optional[List[Tuple[int, Tuple[int, ...], int, tf.DType]]] = None,
    ) -> List[tf.Tensor]:
        """AllReduce all weighted local gradients in one fused MPI buffer.
        
        Formula:
          global_gradient = sum_{i} (local_batch_i * local_grad_i) / global_batch_size
        """
        local_batch = float(self.local_batch_size)
        global_batch = float(self.global_batch_size)

        if pre_packed_parts is not None and pre_grad_metadata is not None:
            flat_parts = pre_packed_parts
            grad_metadata = pre_grad_metadata
        else:
            flat_parts = []
            grad_metadata = []
            for index, local_grad in enumerate(local_grads):
                if local_grad is None:
                    continue
                grad_np = local_grad.numpy().astype(np.float32, copy=False)
                flat_parts.append(grad_np.reshape(-1))
                grad_metadata.append(
                    (index, grad_np.shape, grad_np.size, local_grad.dtype)
                )

        global_grads = [None] * len(local_grads)
        if not flat_parts:
            total_elements = sum(v.shape.num_elements() for v in self.model.trainable_variables)
            fused_local = np.zeros(total_elements, dtype=np.float32)
            fused_sum = np.empty_like(fused_local)
            self.comm.Allreduce(fused_local, fused_sum, op=MPI.SUM)
            fused_global = fused_sum / global_batch
            offset = 0
            for index, v in enumerate(self.model.trainable_variables):
                size = v.shape.num_elements()
                grad_np = fused_global[offset:offset + size].reshape(v.shape)
                global_grads[index] = tf.convert_to_tensor(grad_np, dtype=v.dtype)
                offset += size
            return global_grads

        fused_local = np.concatenate(flat_parts).astype(np.float32, copy=False)
        fused_local *= local_batch
        fused_sum = np.empty_like(fused_local)

        # One blocking collective per training step instead of one per tensor.
        self.comm.Allreduce(fused_local, fused_sum, op=MPI.SUM)
        fused_global = fused_sum / global_batch

        offset = 0
        for index, shape, size, dtype in grad_metadata:
            grad_np = fused_global[offset:offset + size].reshape(shape)
            global_grads[index] = tf.convert_to_tensor(grad_np, dtype=dtype)
            offset += size

        return global_grads

    def verify_gradients(self, global_grads: List[tf.Tensor], step: int) -> float:
        """Verifies that all ranks computed identical averaged global gradients."""
        flat_list = [g.numpy().ravel() for g in global_grads if g is not None]
        flat_grads = np.concatenate(flat_list).astype(np.float32)

        max_buf = np.empty_like(flat_grads)
        min_buf = np.empty_like(flat_grads)

        self.comm.Barrier()
        self.comm.Allreduce(flat_grads, max_buf, op=MPI.MAX)
        self.comm.Allreduce(flat_grads, min_buf, op=MPI.MIN)
        self.comm.Barrier()

        max_diff = float(np.max(np.abs(max_buf - min_buf)))

        if self.rank == 0:
            print(f"[SYNC] step={step} grad_max_diff={max_diff:.8e}", flush=True)

        if max_diff > self.sync_tolerance:
            msg = (
                f"[SYNC FAILED] Gradient sync check failed at step {step}: "
                f"grad_max_diff={max_diff:.8e} > tolerance={self.sync_tolerance}"
            )
            if self.rank == 0:
                print(f"FAILED: {msg}", flush=True)
            raise RuntimeError(msg)

        return max_diff

    def verify_model_weights(self, step: int) -> float:
        """Correction 3: Verifies complete model weights (model.get_weights()) across all ranks."""
        weights = self.model.get_weights()
        flat_weights = np.concatenate([w.ravel() for w in weights]).astype(np.float32)

        max_buf = np.empty_like(flat_weights)
        min_buf = np.empty_like(flat_weights)

        self.comm.Barrier()
        self.comm.Allreduce(flat_weights, max_buf, op=MPI.MAX)
        self.comm.Allreduce(flat_weights, min_buf, op=MPI.MIN)
        self.comm.Barrier()

        max_diff = float(np.max(np.abs(max_buf - min_buf)))

        if self.rank == 0:
            print(f"[SYNC] step={step} weight_max_diff={max_diff:.8e}", flush=True)

        if max_diff > self.sync_tolerance:
            msg = (
                f"[SYNC FAILED] Weight sync check failed at step {step}: "
                f"weight_max_diff={max_diff:.8e} > tolerance={self.sync_tolerance}"
            )
            if self.rank == 0:
                print(f"FAILED: {msg}", flush=True)
            raise RuntimeError(msg)

        return max_diff

    def synchronize_model_delta(self, w_start: List[np.ndarray]) -> Tuple[float, float]:
        """Synchronizes model weights across all ranks using Delta representation (Phase 1 Baseline).

        Formula:
          Delta_i = W_local_i - W_start
          Delta_global = (Delta_0 + Delta_1 + Delta_2 + Delta_3 + Delta_4) / 5
          W_global = W_start + Delta_global

        Returns:
          (sync_time, max_weight_diff)
        """
        t_sync_start = time.perf_counter()

        # 1. Compute local Delta_i = W_local_i - W_start
        w_local = self.model.get_weights()
        delta_i = [wl - ws for wl, ws in zip(w_local, w_start)]

        # 2. Flatten Delta_i for efficient single MPI AllReduce
        flat_delta = np.concatenate([d.ravel() for d in delta_i]).astype(np.float32)
        summed_delta = np.empty_like(flat_delta)

        # 3. MPI AllReduce SUM / 5
        self.comm.Allreduce(flat_delta, summed_delta, op=MPI.SUM)
        global_flat_delta = summed_delta / float(self.world_size)

        # 4. Compute W_global = W_start + Delta_global
        w_global = []
        offset = 0
        for ws in w_start:
            size = ws.size
            chunk = global_flat_delta[offset:offset + size].reshape(ws.shape).astype(ws.dtype)
            w_global.append(ws + chunk)
            offset += size

        # 5. Set W_global to model
        self.model.set_weights(w_global)

        # 6. Verify consistency across all ranks: max_weight_diff
        flat_w = np.concatenate([w.ravel() for w in w_global]).astype(np.float32)
        max_buf = np.empty_like(flat_w)
        min_buf = np.empty_like(flat_w)

        self.comm.Barrier()
        self.comm.Allreduce(flat_w, max_buf, op=MPI.MAX)
        self.comm.Allreduce(flat_w, min_buf, op=MPI.MIN)
        self.comm.Barrier()

        diff_arr = np.abs(max_buf - min_buf)
        max_weight_diff = float(np.max(diff_arr))

        if max_weight_diff > self.sync_tolerance:
            offset = 0
            for var, ws in zip(self.model.weights, w_start):
                size = ws.size
                layer_diff = float(np.max(diff_arr[offset:offset + size]))
                offset += size
                if layer_diff > self.sync_tolerance:
                    print(
                        f"[CONSISTENCY FAILED] Rank {self.rank}: Layer '{var.name}' "
                        f"max difference: {layer_diff:.8e} > {self.sync_tolerance:.1e}",
                        flush=True,
                    )
            raise RuntimeError(
                f"[CONSISTENCY FAILED] Rank {self.rank} weight mismatch: {max_weight_diff:.8e} > {self.sync_tolerance:.1e}"
            )

        # 7. Optimizer state handling
        if self.opt_state_sync == "average":
            for v in self.optimizer.variables():
                if "iteration" not in v.name.lower():
                    val = v.numpy().astype(np.float32)
                    summed_v = np.empty_like(val)
                    self.comm.Allreduce(val, summed_v, op=MPI.SUM)
                    avg_v = (summed_v / float(self.world_size)).astype(v.dtype.as_numpy_dtype)
                    v.assign(avg_v)

        sync_time = time.perf_counter() - t_sync_start
        return sync_time, max_weight_diff

    def _train_local_steps(self):
        """Executes Local-Step Synchronization training loop (Phase 1 Baseline)."""
        import math

        if isinstance(self.local_steps, str) and self.local_steps.lower() == "full_epoch":
            K = self.steps_per_epoch
        else:
            K = int(self.local_steps)
            if K >= self.steps_per_epoch:
                K = self.steps_per_epoch

        total_rounds_per_epoch = math.ceil(self.steps_per_epoch / K)

        if self.rank == 0:
            self.logger.info("=" * 88)
            self.logger.info(" HETEROVIT-MPI: LOCAL-STEP SYNCHRONIZATION BASELINE (PHASE 1)")
            self.logger.info("=" * 88)
            self.logger.info(f"  Sync Mode          : local_steps")
            self.logger.info(f"  Local Steps K      : {K} ({'full_epoch' if K == self.steps_per_epoch else f'{K} steps/round'})")
            self.logger.info(f"  Rounds per Epoch   : {total_rounds_per_epoch}")
            self.logger.info(f"  Steps per Epoch    : {self.steps_per_epoch}")
            opt_desc = (
                "local momentum/variance preserved across rounds"
                if self.opt_state_sync == "preserve_local"
                else "momentum/variance averaged across ranks at sync boundary"
            )
            self.logger.info(f"  Optimizer Policy   : {self.opt_state_sync} ({opt_desc})")
            self.logger.info(f"  Local Batch Size   : {self.local_batch_size} (Global Batch: {self.global_batch_size})")
            self.logger.info(f"  Sync Tolerance     : {self.sync_tolerance:.1e}")
            self.logger.info("=" * 88)

        total_steps_executed = 0
        total_rounds_executed = 0
        train_start_time = time.perf_counter()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.perf_counter()
            total_train_loss = 0.0
            total_train_acc = 0.0
            round_in_epoch = 0
            step_in_epoch = 0

            train_iter = iter(self.dist_train_ds)

            while step_in_epoch < self.steps_per_epoch:
                round_in_epoch += 1
                total_rounds_executed += 1
                target_k = min(K, self.steps_per_epoch - step_in_epoch)

                t_round_start = time.perf_counter()

                # Snapshot W_start at start of round
                w_start = [w.copy() for w in self.model.get_weights()]

                # --- 1. LOCAL STEPS (NO COMMUNICATION) ---
                t_comp_start = time.perf_counter()
                round_loss_sum = 0.0
                round_acc_sum = 0.0
                actual_k = 0

                for _ in range(target_k):
                    step_in_epoch += 1
                    total_steps_executed += 1
                    actual_k += 1
                    try:
                        images, labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.dist_train_ds)
                        images, labels = next(train_iter)

                    loss, acc = self._local_train_step(images, labels)
                    round_loss_sum += float(loss)
                    round_acc_sum += float(acc)

                    if self.max_steps is not None and total_steps_executed >= self.max_steps:
                        break

                compute_time = time.perf_counter() - t_comp_start
                round_loss = round_loss_sum / max(actual_k, 1)
                round_acc = round_acc_sum / max(actual_k, 1)

                total_train_loss += round_loss_sum
                total_train_acc += round_acc_sum

                # --- 2. WAIT TIME MEASUREMENT (BARRIER) ---
                t_wait_start = time.perf_counter()
                self.comm.Barrier()
                wait_time = time.perf_counter() - t_wait_start

                # --- 3. MODEL DELTA MERGE & SET WEIGHTS ---
                sync_time, max_weight_diff = self.synchronize_model_delta(w_start)

                round_time = time.perf_counter() - t_round_start
                samples_processed = actual_k * self.local_batch_size
                round_tput = (actual_k * self.global_batch_size) / max(round_time, 1e-6)

                # Per-rank round metrics gathering
                round_info = {
                    "epoch": epoch,
                    "round": round_in_epoch,
                    "rank": self.rank,
                    "host": self.hostname,
                    "device": self.device_str,
                    "local_steps": actual_k,
                    "batch_size": self.local_batch_size,
                    "samples_processed": samples_processed,
                    "compute_time": compute_time,
                    "wait_time": wait_time,
                    "sync_time": sync_time,
                    "round_time": round_time,
                    "throughput": round_tput,
                    "train_loss": round_loss,
                    "train_acc": round_acc,
                    "max_weight_diff": max_weight_diff,
                }
                all_round_info = self.comm.gather(round_info, root=0)

                if self.rank == 0 and all_round_info:
                    # Write to rounds_timeline.csv
                    if hasattr(self, "rounds_csv") and self.rounds_csv:
                        with open(self.rounds_csv, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            for r_info in sorted(all_round_info, key=lambda x: x["rank"]):
                                writer.writerow([
                                    r_info["epoch"],
                                    r_info["round"],
                                    r_info["rank"],
                                    r_info["local_steps"],
                                    r_info["batch_size"],
                                    r_info["samples_processed"],
                                    f"{r_info['compute_time']:.4f}",
                                    f"{r_info['wait_time']:.4f}",
                                    f"{r_info['sync_time']:.4f}",
                                    f"{r_info['round_time']:.4f}",
                                    f"{r_info['throughput']:.2f}",
                                    f"{r_info['train_loss']:.5f}",
                                ])

                    step_start = step_in_epoch - actual_k + 1
                    step_end = step_in_epoch

                    self.logger.info("=" * 88)
                    self.logger.info(
                        f" [ROUND {round_in_epoch:02d}/{total_rounds_per_epoch:02d} (Epoch {epoch:02d})] "
                        f"K={actual_k} | Steps: {step_start:02d}-{step_end:02d}/{self.steps_per_epoch:02d} | "
                        f"Round Time: {round_time:6.2f}s | Tput: {round_tput:5.1f} img/s | "
                        f"max_weight_diff: {max_weight_diff:.2e}"
                    )
                    self.logger.info("-" * 88)
                    sorted_rounds = sorted(all_round_info, key=lambda x: x["compute_time"])
                    for r_info in sorted_rounds:
                        r = r_info["rank"]
                        h = r_info["host"]
                        dev = r_info["device"]
                        c_s = r_info["compute_time"]
                        w_s = r_info["wait_time"]
                        pct = (w_s / (c_s + w_s) * 100.0) if (c_s + w_s) > 0 else 0.0
                        if w_s <= 0.05:
                            status = f"STRAGGLER (Compute: {c_s:6.2f}s - Bắt cả cụm chờ)"
                        elif r == 0:
                            status = f"Compute: {c_s:6.2f}s | Wait: {w_s:6.2f}s ({pct:4.1f}% idle)"
                        else:
                            status = f"Compute: {c_s:6.2f}s | Wait: {w_s:6.2f}s ({pct:4.1f}% idle)"
                        self.logger.info(f"   * Rank {r} [{h:9s} - {dev:5s}]: {status}")
                    self.logger.info(f"   Sync Time: {sync_time*1000:6.1f}ms | Loss: {round_loss:.4f} | Acc: {round_acc*100:5.2f}%")
                    self.logger.info("=" * 88)

                if self.max_steps is not None and total_steps_executed >= self.max_steps:
                    if self.rank == 0:
                        self.logger.info(f"Reached max_steps={self.max_steps}. Stopping training loop.")
                    break

            # End of epoch calculations
            train_time = time.perf_counter() - epoch_start_time
            avg_train_loss = total_train_loss / max(step_in_epoch, 1)
            avg_train_acc = total_train_acc / max(step_in_epoch, 1)
            total_samples = step_in_epoch * self.global_batch_size
            train_tput = total_samples / max(train_time, 1e-6)

            # Validation phase
            is_smoke = self.max_steps is not None and self.max_steps <= 10
            val_steps_to_run = min(2, self.val_steps) if is_smoke else self.val_steps
            val_start_time = time.perf_counter()
            val_loss, val_acc = self.evaluate(self.dist_val_ds, val_steps_to_run)
            val_time = time.perf_counter() - val_start_time
            total_epoch_time = train_time + val_time

            if self.rank == 0:
                is_best = val_acc > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_acc
                    self.save_checkpoint("best.weights.h5")
                self.save_checkpoint("last.weights.h5")

                try:
                    curr_lr = float(self.optimizer.learning_rate.numpy())
                except Exception:
                    try:
                        curr_lr = float(self.optimizer.learning_rate)
                    except Exception:
                        curr_lr = 1e-3

                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    train_accuracy=avg_train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    epoch_time=total_epoch_time,
                    samples_per_sec=train_tput,
                    train_time=train_time,
                    val_time=val_time,
                    learning_rate=curr_lr,
                    total_epochs=self.epochs,
                    is_best=is_best,
                    best_val_acc=self.best_val_accuracy,
                    cluster_info=f"local_steps (K={K}) | {self.global_batch_size} (128x{self.world_size})",
                )

            import gc
            gc.collect()
            self.comm.Barrier()

            if self.max_steps is not None and total_steps_executed >= self.max_steps:
                break

        self.total_train_time = time.perf_counter() - train_start_time
        if self.rank == 0:
            self.logger.info(
                f"MPI Training finished successfully in {self.total_train_time:.2f}s "
                f"({self.total_train_time / 60:.2f} minutes)."
            )

    def _train_gradient_allreduce(self):
        """Executes synchronous distributed training loop across epochs."""
        if self.rank == 0:
            self.logger.info("Starting Synchronous MPI Training Loop...")
            self.logger.info(f"MPI World Size: {self.world_size} ranks")
            self.logger.info(f"Target Epochs: {self.epochs}")
            if hasattr(self, "rank_batch_sizes") and self.rank_batch_sizes:
                self.logger.info(
                    f"Uneven Workload Batches: {self.rank_batch_sizes} (Global Batch Size: {self.global_batch_size})"
                )
            else:
                self.logger.info(
                    f"Local Batch Size: {self.local_batch_size} (Global Batch Size: {self.global_batch_size})"
                )
            if self.accum_steps > 1:
                self.logger.info(
                    f"Rank 0 Gradient Accumulation: {self.accum_steps} steps of micro_batch={self.micro_batch_size}"
                )
            self.logger.info(f"Steps per epoch: {self.steps_per_epoch}, Val steps: {self.val_steps}")
            if self.max_steps is not None:
                self.logger.info(f"Smoke test mode enabled: terminating after {self.max_steps} steps.")

        total_steps_executed = (self.start_epoch - 1) * self.steps_per_epoch if self.start_epoch > 1 else 0
        train_start_time = time.perf_counter()

        for epoch in range(self.start_epoch, self.epochs + 1):
            epoch_start_time = time.perf_counter()
            total_train_loss = 0.0
            total_train_acc = 0.0
            total_compute_time = 0.0
            step_count = 0

            train_iter = iter(self.dist_train_ds) if (self.dist_train_ds is not None and self.local_batch_size > 0) else None

            for step in range(1, self.steps_per_epoch + 1):
                step_start_time = time.perf_counter()
                total_steps_executed += 1

                # 1. Measure data fetching time (data_ms)
                t_data_start = time.perf_counter()
                if self.local_batch_size > 0 and train_iter is not None:
                    try:
                        images, labels = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.dist_train_ds)
                        images, labels = next(train_iter)
                else:
                    images, labels = None, None
                t_data_end = time.perf_counter()
                data_ms = (t_data_end - t_data_start) * 1000.0

                # 2. Forward pass & local gradient computation (compute_ms)
                # Ensure GPU TensorFlow completes execution before stopping timer (not just host enqueue)
                t_comp_start = time.perf_counter()
                packed_flat_grads = []
                grad_metadata = []
                if self.local_batch_size > 0 and images is not None:
                    loss, acc, local_grads = self._compute_local_grads(images, labels)
                    # Converting gradients to host NumPy array forces GPU stream synchronization (cudaMemcpy),
                    # guaranteeing all forward & backward GPU kernels are fully completed.
                    for idx, local_grad in enumerate(local_grads):
                        if local_grad is not None:
                            grad_np = local_grad.numpy().astype(np.float32, copy=False)
                            packed_flat_grads.append(grad_np.reshape(-1))
                            grad_metadata.append(
                                (idx, grad_np.shape, grad_np.size, local_grad.dtype)
                            )
                else:
                    loss = tf.constant(0.0, dtype=tf.float32)
                    acc = tf.constant(0.0, dtype=tf.float32)
                    local_grads = [tf.zeros_like(v) for v in self.model.trainable_variables]

                t_comp_end = time.perf_counter()
                compute_ms = (t_comp_end - t_comp_start) * 1000.0
                total_compute_time += (t_comp_end - t_comp_start)

                # 3. Synchronization barrier before gradient allreduce (barrier_ms)
                # Measures waiting time of faster ranks waiting for stragglers to reach the sync point
                t_bar_start = time.perf_counter()
                self.comm.Barrier()
                t_bar_end = time.perf_counter()
                barrier_ms = (t_bar_end - t_bar_start) * 1000.0

                # 4. MPI AllReduce SUM -> / global_batch_size (allreduce_ms)
                t_comm_start = time.perf_counter()
                global_grads = self.allreduce_gradients(
                    local_grads,
                    pre_packed_parts=packed_flat_grads,
                    pre_grad_metadata=grad_metadata,
                )
                t_comm_end = time.perf_counter()
                allreduce_ms = (t_comm_end - t_comm_start) * 1000.0

                # Synchronous verification check for gradients (if sync_interval)
                if total_steps_executed % self.sync_interval == 0:
                    self.verify_gradients(global_grads, step=total_steps_executed)

                # 5. Apply globally-averaged gradients (optimizer_ms)
                t_opt_start = time.perf_counter()
                self._apply_global_grads(global_grads)
                if self.rank == 0 and self.num_gpus > 0:
                    # Synchronize optimizer weight updates to GPU memory
                    _ = self.model.trainable_variables[0].numpy()
                t_opt_end = time.perf_counter()
                optimizer_ms = (t_opt_end - t_opt_start) * 1000.0

                # Synchronous verification check for weights (if sync_interval)
                if total_steps_executed % self.sync_interval == 0:
                    self.verify_model_weights(step=total_steps_executed)

                # 6. Step wall time & other_ms
                step_end_time = time.perf_counter()
                step_time = step_end_time - step_start_time
                step_ms = step_time * 1000.0
                accounted_ms = data_ms + compute_ms + allreduce_ms + optimizer_ms + barrier_ms
                other_ms = max(0.0, step_ms - accounted_ms)

                # 7. Gather detailed timing metrics across all ranks
                local_timing = {
                    "epoch": epoch,
                    "step": step,
                    "rank": self.rank,
                    "host": self.hostname,
                    "device": self.device_str,
                    "batch": self.local_batch_size,
                    "data_ms": round(data_ms, 2),
                    "compute_ms": round(compute_ms, 2),
                    "allreduce_ms": round(allreduce_ms, 2),
                    "optimizer_ms": round(optimizer_ms, 2),
                    "barrier_ms": round(barrier_ms, 2),
                    "other_ms": round(other_ms, 2),
                    "step_ms": round(step_ms, 2),
                }
                all_rank_timings = self.comm.gather(local_timing, root=0)

                # 8. Record to CSV and Periodic Rank 0 Terminal Logging
                if self.rank == 0 and all_rank_timings:
                    if self.step_timing_csv:
                        with open(self.step_timing_csv, "a", newline="", encoding="utf-8") as f:
                            csv_w = csv.writer(f)
                            for t_item in all_rank_timings:
                                csv_w.writerow([
                                    t_item["epoch"],
                                    t_item["step"],
                                    t_item["rank"],
                                    t_item["batch"],
                                    f"{t_item['data_ms']:.2f}",
                                    f"{t_item['compute_ms']:.2f}",
                                    f"{t_item['allreduce_ms']:.2f}",
                                    f"{t_item['optimizer_ms']:.2f}",
                                    f"{t_item['barrier_ms']:.2f}",
                                    f"{t_item['other_ms']:.2f}",
                                    f"{t_item['step_ms']:.2f}",
                                ])

                    critical_compute_ms = max(t_item["compute_ms"] for t_item in all_rank_timings)
                    r0_timing = next(t_item for t_item in all_rank_timings if t_item["rank"] == 0)
                    gpu_compute_ms = r0_timing["compute_ms"]
                    gpu_idle_ms = max(0.0, critical_compute_ms - gpu_compute_ms)
                    comm_ratio_pct = (r0_timing["allreduce_ms"] / max(r0_timing["step_ms"], 1e-6)) * 100.0

                    is_smoke = self.max_steps is not None and self.max_steps <= 10
                    is_periodic = (step % self.log_interval == 0 or step == self.steps_per_epoch or is_smoke)

                    if is_periodic:
                        self.logger.info("=" * 102)
                        self.logger.info(
                            f" [STEP TIMING BREAKDOWN - EPOCH {epoch:02d} STEP {step:04d}/{self.steps_per_epoch:04d}]"
                        )
                        self.logger.info(
                            f"   {'Rank':<5} {'Host':<9} {'Device':<6} {'Batch':<6} "
                            f"{'Data(ms)':<9} {'Compute(ms)':<12} {'Barrier(ms)':<12} "
                            f"{'AllReduce(ms)':<14} {'Opt(ms)':<8} {'Other(ms)':<10} {'Step(ms)':<9}"
                        )
                        self.logger.info("   " + "-" * 98)
                        for t_item in sorted(all_rank_timings, key=lambda x: x["rank"]):
                            self.logger.info(
                                f"   {t_item['rank']:<5} {t_item['host']:<9} {t_item['device']:<6} {t_item['batch']:<6} "
                                f"{t_item['data_ms']:>8.2f} {t_item['compute_ms']:>11.2f} {t_item['barrier_ms']:>11.2f} "
                                f"{t_item['allreduce_ms']:>13.2f} {t_item['optimizer_ms']:>7.2f} {t_item['other_ms']:>9.2f} {t_item['step_ms']:>8.2f}"
                            )
                        self.logger.info("   " + "-" * 98)
                        slowest_item = max(all_rank_timings, key=lambda x: x["compute_ms"])
                        self.logger.info(
                            f"   >>> Critical Compute (Slowest Node): {critical_compute_ms:6.2f} ms ({slowest_item['host']} - Rank {slowest_item['rank']})"
                        )
                        self.logger.info(
                            f"   >>> GPU Compute (Rank 0)           : {gpu_compute_ms:6.2f} ms"
                        )
                        self.logger.info(
                            f"   >>> GPU Idle Wait (Compute Imbal.) : {gpu_idle_ms:6.2f} ms "
                            f"({(gpu_idle_ms / critical_compute_ms * 100.0) if critical_compute_ms > 0 else 0.0:4.1f}% of compute)"
                        )
                        self.logger.info(
                            f"   >>> Communication Ratio (AllReduce): {comm_ratio_pct:5.1f}% of total step time "
                            f"({r0_timing['allreduce_ms']:.1f} ms / {r0_timing['step_ms']:.1f} ms)"
                        )
                        self.logger.info("=" * 102)

                # Step-level runtime telemetry recording
                if hasattr(self, "runtime_profiler") and self.runtime_profiler is not None:
                    self.runtime_profiler.record_step(
                        step_id=total_steps_executed,
                        local_batch=self.local_batch_size,
                        compute_start=t_comp_start,
                        compute_end=t_comp_end,
                        sync_enter=t_bar_start,
                        sync_exit=t_bar_end,
                        comm_start=t_comm_start,
                        comm_end=t_comm_end,
                        step_wall_time=step_ms / 1000.0,
                    )

                total_train_loss += float(loss)
                total_train_acc += float(acc)
                step_count += 1

                # Periodic logging on rank 0 (logs every step in smoke test mode)
                is_smoke = self.max_steps is not None and self.max_steps <= 10
                if self.rank == 0 and (step % self.log_interval == 0 or step == self.steps_per_epoch or is_smoke):
                    avg_step_loss = total_train_loss / step_count
                    avg_step_acc = total_train_acc / step_count
                    step_tput = self.global_batch_size / max(step_time, 1e-6)
                    pct = (step / self.steps_per_epoch) * 100.0

                    try:
                        curr_lr = float(self.optimizer.learning_rate.numpy())
                    except Exception:
                        try:
                            curr_lr = float(self.optimizer.learning_rate)
                        except Exception:
                            curr_lr = 1e-3

                    # ETA calculation for remaining steps in this epoch
                    elapsed_epoch = time.perf_counter() - epoch_start_time
                    avg_step_sec = elapsed_epoch / step_count
                    remaining_steps = max(0, self.steps_per_epoch - step)
                    eta_sec = int(remaining_steps * avg_step_sec)
                    eta_m, eta_s = divmod(eta_sec, 60)
                    eta_str = f"{eta_m:02d}:{eta_s:02d}"

                    self.logger.info(
                        f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                        f"[{step:02d}/{self.steps_per_epoch:02d} ({pct:5.1f}%)] | "
                        f"Loss: {float(loss):.4f} (avg: {avg_step_loss:.4f}) | "
                        f"Acc: {float(acc)*100:5.2f}% (avg: {avg_step_acc*100:5.2f}%) | "
                        f"Step: {step_time*1000:6.1f}ms | "
                        f"Tput: {step_tput:5.1f} img/s | "
                        f"LR: {curr_lr:.2e} | "
                        f"ETA: {eta_str}"
                    )

                if self.max_steps is not None and total_steps_executed >= self.max_steps:
                    if self.rank == 0:
                        self.logger.info(f"Reached max_steps={self.max_steps}. Stopping training loop.")
                    break

                if step >= self.steps_per_epoch:
                    break

            # End of epoch metrics
            train_time = time.perf_counter() - epoch_start_time
            total_samples = step_count * self.global_batch_size
            train_tput = total_samples / max(train_time, 1e-6)
            avg_train_loss = total_train_loss / max(step_count, 1)
            avg_train_acc = total_train_acc / max(step_count, 1)

            # Per-rank performance gathering across all ranks
            avg_comp_ms = (total_compute_time / max(step_count, 1)) * 1000.0
            epoch_timing = {
                "rank": self.rank,
                "host": self.hostname,
                "device": self.device_str,
                "batch": self.local_batch_size,
                "local_loss": avg_train_loss,
                "local_acc": avg_train_acc,
                "avg_compute_ms": avg_comp_ms,
            }
            all_epoch_timings = self.comm.gather(epoch_timing, root=0)

            if self.rank == 0 and all_epoch_timings:
                slowest_avg_ms = max(t["avg_compute_ms"] for t in all_epoch_timings)
                fastest_avg_ms = min(t["avg_compute_ms"] for t in all_epoch_timings)
                avg_speedup = slowest_avg_ms / max(fastest_avg_ms, 1e-6)

                self.logger.info("-" * 88)
                self.logger.info(f" [PER-RANK PERFORMANCE BREAKDOWN - EPOCH {epoch:03d}]")
                sorted_epoch_timings = sorted(all_epoch_timings, key=lambda x: x["avg_compute_ms"])
                for rank_info in sorted_epoch_timings:
                    r = rank_info["rank"]
                    h = rank_info["host"]
                    dev = rank_info["device"]
                    b = rank_info.get("batch", self.local_batch_size)
                    l = rank_info["local_loss"]
                    a = rank_info["local_acc"]
                    c_ms = rank_info["avg_compute_ms"]
                    w_ms = slowest_avg_ms - c_ms
                    idle_pct = (w_ms / slowest_avg_ms * 100.0) if slowest_avg_ms > 0 else 0.0
                    is_straggler = "STRAGGLER" if w_ms <= 1.0 else ("GPU WAITING" if r == 0 else "WAITING")

                    self.logger.info(
                        f"   Rank {r} [{h:9s} - {dev:5s} (batch={b:<3d})]: "
                        f"Loss: {l:.4f} | Acc: {a*100:5.2f}% | "
                        f"Compute: {c_ms:7.1f}ms | Idle Wait: {w_ms:7.1f}ms ({idle_pct:4.1f}% idle) [{is_straggler}]"
                    )

                    # Save to CSV
                    if hasattr(self, "timeline_csv") and self.timeline_csv:
                        with open(self.timeline_csv, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                epoch, r, h, dev, b, f"{l:.5f}", f"{a:.5f}",
                                f"{c_ms:.2f}", f"{w_ms:.2f}", f"{idle_pct:.2f}", is_straggler
                            ])

                self.logger.info(
                    f"   >>> Tóm tắt Epoch: GPU (Rank 0) nhanh gấp {avg_speedup:.1f}x CPU chậm nhất; "
                    f"GPU lãng phí {(1.0 - fastest_avg_ms / slowest_avg_ms)*100.0:.1f}% thời gian do chờ đợi."
                )
                self.logger.info("-" * 88)

            # Validation phase
            is_smoke = self.max_steps is not None and self.max_steps <= 10
            val_steps_to_run = min(2, self.val_steps) if is_smoke else self.val_steps
            val_start_time = time.perf_counter()
            val_loss, val_acc = self.evaluate(self.dist_val_ds, val_steps_to_run)
            val_time = time.perf_counter() - val_start_time
            total_epoch_time = train_time + val_time

            if self.rank == 0:
                is_best = val_acc > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_acc
                    self.save_checkpoint("best.weights.h5")
                self.save_checkpoint("last.weights.h5")

                try:
                    curr_lr = float(self.optimizer.learning_rate.numpy())
                except Exception:
                    try:
                        curr_lr = float(self.optimizer.learning_rate)
                    except Exception:
                        curr_lr = 1e-3

                batch_summary = (
                    f"{self.global_batch_size} (uneven {self.rank_batch_sizes})"
                    if hasattr(self, "rank_batch_sizes") and self.rank_batch_sizes
                    else f"{self.global_batch_size} ({self.local_batch_size} x {self.world_size} ranks)"
                )

                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    train_accuracy=avg_train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    epoch_time=total_epoch_time,
                    samples_per_sec=train_tput,
                    train_time=train_time,
                    val_time=val_time,
                    learning_rate=curr_lr,
                    total_epochs=self.epochs,
                    is_best=is_best,
                    best_val_acc=self.best_val_accuracy,
                    cluster_info=batch_summary,
                )

            # --- Phase 2: Closed-Loop Dynamic Workload Rebalancer ---
            if self.dynamic_rebalance_enabled and epoch < self.epochs:
                # 1. Gather rich runtime telemetry from each worker's RuntimeProfiler
                if hasattr(self, "runtime_profiler") and self.runtime_profiler is not None:
                    telemetry = self.runtime_profiler.get_latest_telemetry(current_batch=self.local_batch_size)
                    telemetry["node_id"] = self.node_id
                    telemetry["rank"] = self.rank
                else:
                    telemetry = {
                        "rank": self.rank,
                        "node_id": self.node_id,
                        "current_batch": self.local_batch_size,
                        "compute_ms": avg_comp_ms,
                        "compute_ema_ms": avg_comp_ms,
                        "runtime_std_ms": 10.0,
                        "comm_ms": 1600.0,
                        "idle_ms": 0.0,
                        "active": self.local_batch_size > 0,
                    }
                all_telemetry = self.comm.gather(telemetry, root=0)

                rebalance_action = "KEEP"
                target_batches = list(self.rank_batch_sizes)

                if self.rank == 0 and self.dynamic_rebalancer is not None and all_telemetry:
                    current_alloc = {
                        self.rank_to_node.get(r, f"lab0{r+1}"): self.rank_batch_sizes[r]
                        for r in range(self.world_size)
                    }
                    decision = self.dynamic_rebalancer.decide_rebalance(
                        epoch=epoch,
                        telemetry_list=all_telemetry,
                        current_allocation=current_alloc,
                        global_batch=self.global_batch_size,
                        measured_t_critical_ms=slowest_avg_ms + 1600.0,
                    )
                    rebalance_action = decision.action
                    target_alloc = decision.target_allocation
                    target_batches = [
                        target_alloc.get(self.rank_to_node.get(r, f"lab0{r+1}"), self.rank_batch_sizes[r])
                        for r in range(self.world_size)
                    ]

                    if decision.action == "SWITCH":
                        self.logger.info("=" * 88)
                        self.logger.info(f" [DYNAMIC REBALANCER - EPOCH {epoch:03d} WORKLOAD REBALANCED]")
                        self.logger.info(f"   Old Allocation : {decision.current_allocation}")
                        self.logger.info(f"   New Allocation : {decision.target_allocation}")
                        self.logger.info(f"   Predicted Gain : {decision.predicted_gain_pct:.1f}%")
                        self.logger.info(f"   Reason         : {decision.reason}")
                        self.logger.info(f"   Overhead       : {decision.scheduler_overhead_ms:.2f} ms")
                        self.logger.info("=" * 88)
                    else:
                        self.logger.info(
                            f" [DYNAMIC REBALANCER - EPOCH {epoch:03d}] Action: {decision.action} | Reason: {decision.reason}"
                        )

                rebalance_msg = {
                    "action": rebalance_action,
                    "target_batches": target_batches,
                } if self.rank == 0 else None
                rebalance_msg = self.comm.bcast(rebalance_msg, root=0)

                if rebalance_msg["action"] == "SWITCH":
                    new_b = rebalance_msg["target_batches"]
                    if new_b != self.rank_batch_sizes:
                        self.rebuild_dataset(new_b)
                        if self.rank == 0:
                            self.logger.info(
                                f"   >>> Successfully re-batched cluster! New rank batches: {self.rank_batch_sizes} "
                                f"(Global: {self.global_batch_size}, Steps/epoch: {self.steps_per_epoch})"
                            )

            # Force garbage collection across all ranks (vital for 4GB node iciplab03)
            import gc
            gc.collect()

            self.comm.Barrier()

            if self.max_steps is not None and total_steps_executed >= self.max_steps:
                break

        self.total_train_time = time.perf_counter() - train_start_time
        if self.rank == 0:
            self.logger.info(
                f"MPI Training finished successfully in {self.total_train_time:.2f}s "
                f"({self.total_train_time / 60:.2f} minutes)."
            )

    def train(self):
        """Executes training loop based on configured sync_mode."""
        if self.sync_mode == "local_steps":
            self._train_local_steps()
        else:
            self._train_gradient_allreduce()

    def evaluate(self, dataset, steps: int) -> Tuple[float, float]:
        """Evaluates model performance on validation/test dataset."""
        if self.rank == 0 and self.strategy is not None:
            global_batch = int(self.local_batch_size)

            def val_step_fn(images, labels):
                predictions = self.model(images, training=False)
                per_example_loss = self.loss_fn(labels, predictions)
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch)
                pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                return loss, accuracy

            @tf.function
            def dist_val_step(images, labels):
                per_replica_losses, per_replica_accs = self.strategy.run(val_step_fn, args=(images, labels))
                total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
                total_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
                return total_loss, total_acc

            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

            for images, labels in dataset:
                loss, acc = dist_val_step(images, labels)
                total_loss += float(loss)
                total_acc += float(acc)
                num_batches += 1
                if steps is not None and num_batches >= steps:
                    break

            avg_loss = total_loss / max(num_batches, 1)
            avg_acc = total_acc / max(num_batches, 1)
            return avg_loss, avg_acc
        else:
            total_loss = 0.0
            total_acc = 0.0
            num_batches = 0

            @tf.function
            def cpu_val_step(images, labels):
                predictions = self.model(images, training=False)
                loss = self.loss_fn(labels, predictions)
                pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
                accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
                return loss, accuracy

            for images, labels in dataset:
                loss, acc = cpu_val_step(images, labels)
                total_loss += float(loss)
                total_acc += float(acc)
                num_batches += 1
                if steps is not None and num_batches >= steps:
                    break

            avg_loss = total_loss / max(num_batches, 1)
            avg_acc = total_acc / max(num_batches, 1)
            return avg_loss, avg_acc

