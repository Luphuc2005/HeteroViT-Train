"""Experiment logger, CSV metric recorder, and run directory manager."""
import os
import csv
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import yaml

from src.metrics.system_metrics import collect_run_metadata

# Vietnam Timezone (UTC+7, Asia/Ho_Chi_Minh)
VN_TZ = timezone(timedelta(hours=7))


class VietnamTimeFormatter(logging.Formatter):
    """Custom formatter ensuring log timestamps always reflect Vietnam Time (UTC+7)."""

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=VN_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S")


class ExperimentLogger:
    """Manages experiment directory structure, console/file logging, and CSV metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.exp_name = config.get("experiment", {}).get("name", "experiment")
        base_output_dir = config.get("logging", {}).get("output_dir", "./results")

        # Generate unique timestamped directory in Vietnam Time: results/<exp_name>_YYYYMMDD_HHMMSS
        timestamp = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_output_dir, f"{self.exp_name}_{timestamp}")
        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # File paths
        self.log_file = os.path.join(self.run_dir, "run.log")
        self.csv_file = os.path.join(self.run_dir, "train.csv")
        self.metadata_file = os.path.join(self.run_dir, "metadata.json")
        self.config_copy_file = os.path.join(self.run_dir, "config.yaml")

        self._setup_logger()
        self._save_config_and_metadata()
        self._init_csv()

    def _setup_logger(self):
        """Sets up Python logger with both StreamHandler and FileHandler using Vietnam Time."""
        self.logger = logging.getLogger(f"Exp_{self.exp_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = VietnamTimeFormatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        fh = logging.FileHandler(self.log_file, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

    def _save_config_and_metadata(self):
        """Saves a replica of configuration YAML and system environment metadata."""
        # Save config copy
        with open(self.config_copy_file, "w", encoding="utf-8") as f:
            yaml.dump(dict(self.config), f, default_flow_style=False)

        # Collect and save metadata
        metadata = collect_run_metadata(self.config)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Initialized experiment run directory: {self.run_dir}")
        self.logger.info(f"Hostname: {metadata['hostname']}")
        self.logger.info(f"Python: {metadata['python_version']} | TensorFlow: {metadata['tensorflow_version']}")
        self.logger.info(f"Allocated CPU cores: {metadata['cpu_cores_allocated']} | Affinity: {metadata['cpu_affinity']}")
        self.logger.info(f"GPU Info: {metadata['gpu_info']}")

    def _init_csv(self):
        """Initializes CSV file with standard headers."""
        headers = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "epoch_time",
            "samples_per_sec",
        ]
        with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_accuracy: float,
        val_loss: float,
        val_accuracy: float,
        epoch_time: float,
        samples_per_sec: float,
    ):
        """Logs metrics for an epoch to console, log file, and CSV."""
        # Write to CSV
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.5f}",
                f"{train_accuracy:.5f}",
                f"{val_loss:.5f}",
                f"{val_accuracy:.5f}",
                f"{epoch_time:.2f}",
                f"{samples_per_sec:.2f}",
            ])

        # Write to log
        msg = (
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} - Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy * 100:.2f}% | "
            f"Time: {epoch_time:.2f}s ({samples_per_sec:.1f} samples/s)"
        )
        self.logger.info(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)
