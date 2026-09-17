"""GPU-only Trainer optimized for 1 V100 GPU baseline."""
import time
import tensorflow as tf
from src.training.base_trainer import BaseTrainer, train_step, get_optimizer



def configure_gpu_runtime():
    """Configures GPU memory growth to prevent TensorFlow from allocating all VRAM at once."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError(
            "GPU mode selected but no physical GPU detected by TensorFlow. "
            "Please check CUDA_VISIBLE_DEVICES and NVIDIA driver installation."
        )
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[Warning] Failed setting memory growth: {e}")


class GPUTrainer(BaseTrainer):
    """Trainer implementation dedicated to 1 V100 GPU execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configure_gpu_runtime()

    def train(self):
        self.logger.info("Starting GPU Training Loop on 1 V100...")
        self.logger.info(f"Target Epochs: {self.epochs}, Batch Size: {self.batch_size}")
        self.logger.info(f"Steps per epoch: {self.steps_per_epoch}, Val steps: {self.val_steps}")

        with tf.device("/GPU:0"):
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

        self.logger.info("GPU Training finished successfully.")


class MultiGPUTrainer(BaseTrainer):
    """Trainer implementation dedicated to Multi-GPU execution via tf.distribute.MirroredStrategy."""

    def __init__(self, strategy: tf.distribute.Strategy, *args, **kwargs):
        self.strategy = strategy
        super().__init__(*args, **kwargs)

        # Re-initialize optimizer and loss function inside strategy scope
        with self.strategy.scope():
            training_cfg = self.config.get("training", {})
            self.optimizer = get_optimizer(training_cfg)
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE,
            )

        # Distribute datasets
        self.dist_train_ds = self.strategy.experimental_distribute_dataset(self.train_ds)
        self.dist_val_ds = self.strategy.experimental_distribute_dataset(self.val_ds)

    def train(self):
        num_replicas = self.strategy.num_replicas_in_sync
        self.logger.info(f"Starting Multi-GPU Training Loop on {num_replicas} GPUs...")
        self.logger.info(f"Target Epochs: {self.epochs}, Global Batch Size: {self.batch_size}")
        self.logger.info(f"Per-GPU Local Batch Size: {self.batch_size // num_replicas}")
        self.logger.info(f"Steps per epoch: {self.steps_per_epoch}, Val steps: {self.val_steps}")

        global_batch = float(self.batch_size)

        def step_fn(images, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                per_example_loss = self.loss_fn(labels, predictions)
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))
            return loss, accuracy

        @tf.function
        def dist_train_step(images, labels):
            per_replica_losses, per_replica_accs = self.strategy.run(step_fn, args=(images, labels))
            total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
            total_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accs, axis=None)
            return total_loss, total_acc

        for epoch in range(1, self.epochs + 1):
            epoch_start_time = time.perf_counter()
            total_train_loss = 0.0
            total_train_acc = 0.0
            total_samples = 0
            step_count = 0

            for step, (images, labels) in enumerate(self.dist_train_ds.take(self.steps_per_epoch), start=1):
                step_start_time = time.perf_counter()
                loss, acc = dist_train_step(images, labels)
                step_time = time.perf_counter() - step_start_time

                total_train_loss += float(loss)
                total_train_acc += float(acc)
                total_samples += self.batch_size
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
            val_loss, val_acc = self.evaluate(self.dist_val_ds, self.val_steps)

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

        self.logger.info(f"Multi-GPU Training on {num_replicas} GPUs finished successfully.")

    def evaluate(self, dataset, steps: int):
        if not isinstance(dataset, tf.distribute.DistributedDataset):
            dataset = self.strategy.experimental_distribute_dataset(dataset)

        global_batch = float(self.batch_size)

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

        for images, labels in dataset.take(steps):
            loss, acc = dist_val_step(images, labels)
            total_loss += float(loss)
            total_acc += float(acc)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        return avg_loss, avg_acc

