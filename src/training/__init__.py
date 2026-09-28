"""Training modules for heterogeneous experimentation.

Imports are lazy so multiprocessing workers can configure CUDA visibility and
CPU affinity before TensorFlow is loaded.
"""

__all__ = [
    "BaseTrainer",
    "train_step",
    "val_step",
    "get_optimizer",
    "CPUTrainer",
    "GPUTrainer",
    "JointTrainer",
]


def __getattr__(name):
    if name in {"BaseTrainer", "train_step", "val_step", "get_optimizer"}:
        from src.training import base_trainer

        return getattr(base_trainer, name)
    if name == "CPUTrainer":
        from src.training.trainer_cpu import CPUTrainer

        return CPUTrainer
    if name == "GPUTrainer":
        from src.training.trainer_gpu import GPUTrainer

        return GPUTrainer
    if name == "JointTrainer":
        from src.training.trainer_joint import JointTrainer

        return JointTrainer
    raise AttributeError(name)
