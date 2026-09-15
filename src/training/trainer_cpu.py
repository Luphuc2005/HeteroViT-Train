"""CPU-only Trainer configured for multi-core benchmarks with thread affinity."""
import os
import time
import tensorflow as tf
from src.training.base_trainer import BaseTrainer, train_step


def configure_cpu_runtime(cpu_cfg: dict):
    """Sets CPU threading and thread affinity controls for TensorFlow."""
    intra_threads = int(cpu_cfg.get("intra_op_threads", 12))
    inter_threads = int(cpu_cfg.get("inter_op_threads", 2))

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        print(f"[Info] TensorFlow CPU threading initialized: intra_op={intra_threads}, inter_op={inter_threads}")
    except RuntimeError as e:
        # Threads already initialized
        print(f"[Warning] Failed setting TF threads: {e}")


class CPUTrainer(BaseTrainer):
    """Trainer implementation dedicated to CPU-only execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: configure_cpu_runtime is already performed early in train.py before any TF ops

    def train(self):
        self.logger.info("Starting CPU Training Loop...")
        self.logger.info(f"Target Epochs: {self.epochs}, Batch Size: {self.batch_size}")
        self.logger.info(f"Steps per epoch: {self.steps_per_epoch}, Val steps: {self.val_steps}")

        with tf.device("/CPU:0"):
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.perf_counter()
                total_train_loss = 0.0
                total_train_acc = 0.0
                total_samples = 0
                step_count = 0

                for step, (images, labels) in enumerate(self.train_ds.take(self.steps_per_epoch), start=1):
                    step_start_time = time.perf_counter()
                    batch_size = tf.shape(images)[0]

                    loss, acc, _ = train_step(
                        self.model,
                        self.optimizer,
                        images,
                        labels,
                        self.loss_fn,
                    )

                    step_time = time.perf_counter() - step_start_time
                    total_train_loss += float(loss)
                    total_train_acc += float(acc)
                    total_samples += int(batch_size)
                    step_count += 1

                    if step % self.log_interval == 0 or step == self.steps_per_epoch:
                        avg_step_loss = total_train_loss / step_count
                        avg_step_acc = total_train_acc / step_count
                        self.logger.info(
                            f"Epoch [{epoch:03d}/{self.epochs:03d}] "
                            f"Step [{step:04d}/{self.steps_per_epoch:04d}] - "
                            f"Loss: {avg_step_loss:.4f} - "
                            f"Acc: {avg_step_acc * 100:.2f}% - "
                            f"Step Time: {step_time * 1000:.1f}ms"
                        )

                epoch_time = time.perf_counter() - epoch_start_time
                samples_per_sec = total_samples / max(epoch_time, 1e-6)
                avg_train_loss = total_train_loss / max(step_count, 1)
                avg_train_acc = total_train_acc / max(step_count, 1)

                # Validation phase
                val_loss, val_acc = self.evaluate(self.val_ds, self.val_steps)

                # Log metrics to CSV and file
                self.logger.log_epoch(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    train_accuracy=avg_train_acc,
                    val_loss=val_loss,
                    val_accuracy=val_acc,
                    epoch_time=epoch_time,
                    samples_per_sec=samples_per_sec,
                )

                # Save checkpoints
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.save_checkpoint("best.weights.h5")
                    self.logger.info(f"New best validation accuracy: {val_acc * 100:.2f}%. Saved best.weights.h5")

                self.save_checkpoint("last.weights.h5")

        self.logger.info("CPU Training finished successfully.")
