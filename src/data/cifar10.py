"""CIFAR-10 data loader with offline data validation and efficient tf.data pipeline."""
import os
import pickle
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf

from src.data.augmentation import get_train_augmentation, get_val_augmentation


def _load_cifar_batch(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads a single CIFAR-10 python batch file."""
    with open(file_path, "rb") as f:
        data_dict = pickle.load(f, encoding="bytes")
        images = data_dict[b"data"]
        labels = data_dict[b"labels"]
        # Reshape (N, 3, 32, 32) -> (N, 32, 32, 3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(labels, dtype=np.int32)
        return images, labels


def load_cifar10_raw(data_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads raw CIFAR-10 arrays from local disk.

    Supports either:
      1. Official Python batch directory structure:
         data_batch_1 ... data_batch_5 and test_batch
         (located directly in data_dir or in data_dir/cifar-10-batches-py)
      2. NumPy archive file: cifar10.npz in data_dir

    Raises:
        FileNotFoundError: If the dataset files cannot be located.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"CIFAR-10 dataset not found at {data_dir}.\n"
            f"Please prepare the dataset before training."
        )

    # Check for cifar10.npz
    npz_path = os.path.join(data_dir, "cifar10.npz")
    if os.path.isfile(npz_path):
        with np.load(npz_path) as data:
            x_train = data["x_train"]
            y_train = data["y_train"]
            x_test = data["x_test"]
            y_test = data["y_test"]
            return x_train, y_train.squeeze(), x_test, y_test.squeeze()

    # Check for python batch files
    target_dir = data_dir
    batches_sub = os.path.join(data_dir, "cifar-10-batches-py")
    if os.path.isdir(batches_sub):
        target_dir = batches_sub

    batch_files = [os.path.join(target_dir, f"data_batch_{i}") for i in range(1, 6)]
    test_file = os.path.join(target_dir, "test_batch")

    all_exist = all(os.path.isfile(bf) for bf in batch_files) and os.path.isfile(test_file)

    if not all_exist:
        raise FileNotFoundError(
            f"CIFAR-10 dataset not found at {data_dir}.\n"
            f"Please prepare the dataset before training."
        )

    # Load 5 training batches
    train_images_list, train_labels_list = [], []
    for bf in batch_files:
        imgs, lbls = _load_cifar_batch(bf)
        train_images_list.append(imgs)
        train_labels_list.append(lbls)

    x_train = np.concatenate(train_images_list, axis=0)
    y_train = np.concatenate(train_labels_list, axis=0)

    # Load test batch
    x_test, y_test = _load_cifar_batch(test_file)

    return x_train, y_train, x_test, y_test


def build_cifar10_datasets(
    data_dir: str,
    batch_size: int = 128,
    val_split: float = 0.1,
    image_size: int = 32,
    seed: int = 42,
    cache: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int, int]:
    """Prepares train, validation, and test tf.data pipelines for CIFAR-10.

    Args:
        data_dir: Directory where CIFAR-10 dataset is located.
        batch_size: Batch size for training and evaluation.
        val_split: Fraction of training data to use for validation (default 0.1 -> 5000 samples).
        image_size: Target image dimensions (32 for standard CIFAR-10).
        seed: Random seed for splitting and shuffling.
        cache: Whether to cache datasets in memory.

    Returns:
        (train_ds, val_ds, test_ds, steps_per_epoch, val_steps)
    """
    x_train_all, y_train_all, x_test, y_test = load_cifar10_raw(data_dir)

    num_total_train = len(x_train_all)
    num_val = int(num_total_train * val_split)

    # Deterministic split using numpy permutation
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_total_train)
    val_idx = indices[:num_val]
    train_idx = indices[num_val:]

    x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
    x_val, y_val = x_train_all[val_idx], y_train_all[val_idx]

    num_train = len(x_train)
    steps_per_epoch = int(np.ceil(num_train / batch_size))
    val_steps = int(np.ceil(num_val / batch_size))

    # Augmentations
    train_aug = get_train_augmentation(image_size)
    val_aug = get_val_augmentation(image_size)

    # Build Train Dataset: raw -> cache -> shuffle -> random augmentation -> batch -> prefetch
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if cache:
        train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=min(num_train, 10000), seed=seed)
    train_ds = train_ds.map(train_aug, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=False)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Build Validation Dataset
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(val_aug, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False)
    if cache:
        val_ds = val_ds.cache()
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Build Test Dataset
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(val_aug, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=False)
    if cache:
        test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, steps_per_epoch, val_steps
