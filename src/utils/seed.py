"""Reproducibility utilities: setting random seeds across Python, NumPy, and TensorFlow."""
import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Delay import of tensorflow so modules can be imported without immediate TF initialization
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Enforce deterministic ops if requested / supported
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass
