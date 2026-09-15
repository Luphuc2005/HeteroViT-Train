"""Base Trainer with custom training loop, gradient extraction, and metric tracking."""
import os
import time
from typing import Dict, Any, Tuple
import tensorflow as tf
from src.metrics.logger import ExperimentLogger


def get_optimizer(training_cfg: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
    """Builds optimizer from config."""
    opt_name = training_cfg.get("optimizer", "adamw").lower()
    lr = float(training_cfg.get("learning_rate", 1e-3))
    wd = float(training_cfg.get("weight_decay", 1e-4))

    if opt_name == "adamw":
        if hasattr(tf.keras.optimizers, "AdamW"):
            return tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        elif hasattr(tf.keras.optimizers, "experimental") and hasattr(tf.keras.optimizers.experimental, "AdamW"):
            return tf.keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=wd)
        else:
            try:
                import tensorflow_addons as tfa
                return tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
            except ImportError:
                return tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


@tf.function
def train_step(
    model: tf.keras.Model, # model ViT
    optimizer: tf.keras.optimizers.Optimizer,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_fn: tf.keras.losses.Loss,
) -> Tuple[tf.Tensor, tf.Tensor, list]:
    """Single training step returning loss, accuracy, and gradients.

    Designed for future extension:
      - Local worker loss
      - Local gradients extraction for aggregation
      - Heterogeneous workload scheduling

    Args:
        model: ViT model
        optimizer: Keras optimizer
        images: Input batch [B, H, W, C]
        labels: Label batch [B]
        loss_fn: Loss function (SparseCategoricalCrossentropy)

    Returns:
        loss: Batch loss value
        accuracy: Batch accuracy value
        gradients: List of computed gradients corresponding to model.trainable_variables
    """
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Calculate batch accuracy
    pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))

    return loss, accuracy, gradients


@tf.function
def val_step(
    model: tf.keras.Model,
    images: tf.Tensor,
    labels: tf.Tensor,
    loss_fn: tf.keras.losses.Loss,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Validation step without gradient computation."""
    predictions = model(images, training=False)
    loss = loss_fn(labels, predictions)

    pred_labels = tf.argmax(predictions, axis=-1, output_type=labels.dtype)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_labels, labels), tf.float32))

    return loss, accuracy


class BaseTrainer:
    """Base trainer managing common epoch iteration, evaluation, metrics, and checkpointing."""

    def __init__(
        self,
        model: tf.keras.Model,
        config: Dict[str, Any],
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        test_ds: tf.data.Dataset,
        steps_per_epoch: int,
        val_steps: int,
        logger: ExperimentLogger,
    ):
        self.model = model
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps
        self.logger = logger

        training_cfg = config.get("training", {})
        self.epochs = int(training_cfg.get("epochs", 100))
        self.batch_size = int(training_cfg.get("batch_size", 128))
        self.log_interval = int(config.get("logging", {}).get("log_interval", 50))

        self.optimizer = get_optimizer(training_cfg)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.best_val_accuracy = 0.0

    def evaluate(self, dataset: tf.data.Dataset, steps: int) -> Tuple[float, float]:
        """Evaluates model performance on dataset."""
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for images, labels in dataset.take(steps):
            loss, acc = val_step(self.model, images, labels, self.loss_fn)
            total_loss += float(loss)
            total_acc += float(acc)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        return avg_loss, avg_acc

    def save_checkpoint(self, filename: str):
        """Saves model weights to checkpoints directory."""
        ckpt_path = os.path.join(self.logger.checkpoints_dir, filename)
        self.model.save_weights(ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")

    def train(self):
        """Main training loop across epochs."""
        raise NotImplementedError("Subclasses must implement train()")
