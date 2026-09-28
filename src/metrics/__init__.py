"""Metrics and logging module with dependency-light lazy imports."""

__all__ = [
    "ExperimentLogger",
    "collect_run_metadata",
    "get_cpu_affinity",
    "get_gpu_info",
]


def __getattr__(name):
    if name == "ExperimentLogger":
        from src.metrics.logger import ExperimentLogger

        return ExperimentLogger
    if name in {"collect_run_metadata", "get_cpu_affinity", "get_gpu_info"}:
        from src.metrics import system_metrics

        return getattr(system_metrics, name)
    raise AttributeError(name)
