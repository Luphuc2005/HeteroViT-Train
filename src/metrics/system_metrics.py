"""System and hardware metrics collection for experimental reproducibility."""
import os
import sys
import socket
import platform
import subprocess
from typing import Dict, Any


def get_cpu_affinity() -> Dict[str, Any]:
    """Retrieves current process CPU affinity (Linux taskset or sched_getaffinity)."""
    affinity_info = {
        "cores_allocated_count": None,
        "affinity_mask": None,
        "total_system_cpus": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(list(os.sched_getaffinity(0)))
            affinity_info["cores_allocated_count"] = len(affinity)
            affinity_info["affinity_mask"] = affinity
        except Exception as e:
            affinity_info["affinity_mask"] = f"Error querying affinity: {e}"
    else:
        affinity_info["affinity_mask"] = "sched_getaffinity not available on this OS"
        affinity_info["cores_allocated_count"] = os.cpu_count()

    return affinity_info


def get_gpu_info() -> Dict[str, Any]:
    """Retrieves GPU device details if available."""
    gpu_info = {
        "gpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"),
    }

    try:
        import tensorflow as tf
        physical_gpus = tf.config.list_physical_devices("GPU")
        gpu_info["gpu_count"] = len(physical_gpus)
        gpu_info["gpu_available"] = len(physical_gpus) > 0

        # Try getting device name via nvidia-smi
        if gpu_info["gpu_available"] or (os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ["", "-1"]):
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                lines = [line.strip() for line in res.stdout.strip().split("\n") if line.strip()]
                gpu_info["gpu_names"] = lines
    except Exception:
        pass

    return gpu_info


def collect_run_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collects system environment and experiment metadata."""
    import tensorflow as tf

    cpu_affinity = get_cpu_affinity()
    gpu_info = get_gpu_info()

    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})

    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cpu_cores_allocated": cpu_affinity["cores_allocated_count"],
        "cpu_affinity": cpu_affinity["affinity_mask"],
        "total_system_cpus": cpu_affinity["total_system_cpus"],
        "gpu_info": gpu_info,
        "batch_size": training_cfg.get("batch_size"),
        "learning_rate": training_cfg.get("learning_rate"),
        "seed": config.get("seed", 42),
        "mode": config.get("mode", "unknown"),
        "model_configuration": dict(model_cfg),
        "cpu_configuration": dict(config.get("cpu", {})),
    }

    return metadata
