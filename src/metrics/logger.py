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

    def __init__(self, config: Dict[str, Any], resume_dir: str = None):
        self.config = config
        self.exp_name = config.get("experiment", {}).get("name", "experiment")
        base_output_dir = os.environ.get("OUTPUT_DIR", config.get("logging", {}).get("output_dir", "./results"))

        if resume_dir and os.path.exists(resume_dir):
            self.run_dir = resume_dir
            self.is_resumed = True
        else:
            # Generate unique timestamped directory in Vietnam Time: results/<exp_name>_YYYYMMDD_HHMMSS
            timestamp = datetime.now(VN_TZ).strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(base_output_dir, f"{self.exp_name}_{timestamp}")
            self.is_resumed = False

        self.checkpoints_dir = os.path.join(self.run_dir, "checkpoints")

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # File paths
        self.log_file = os.path.join(self.run_dir, "run.log")
        self.csv_file = os.path.join(self.run_dir, "train.csv")
        self.metadata_file = os.path.join(self.run_dir, "metadata.json")
        self.config_copy_file = os.path.join(self.run_dir, "config.yaml")

        self._setup_logger()
        if not self.is_resumed:
            self._save_config_and_metadata()
            self._init_csv()
        else:
            self.logger.info("=" * 88)
            self.logger.info(f" [RESUME] Resuming training in existing directory: {self.run_dir}")
            self.logger.info("=" * 88)

    def _setup_logger(self):
        """Sets up Python logger with both StreamHandler and FileHandler using Vietnam Time."""
        self.logger = logging.getLogger(f"Exp_{self.exp_name}_{os.path.basename(self.run_dir)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Clear existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = VietnamTimeFormatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler: append mode if resuming, write mode if fresh run
        file_mode = "a" if self.is_resumed else "w"
        fh = logging.FileHandler(self.log_file, mode=file_mode, encoding="utf-8")
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
        """Initializes CSV file with standard publication-grade headers."""
        headers = [
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "epoch_time",
            "train_time",
            "val_time",
            "samples_per_sec",
            "learning_rate",
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
        train_time: float = None,
        val_time: float = None,
        learning_rate: float = None,
        total_epochs: int = None,
        is_best: bool = False,
        best_val_acc: float = None,
        cluster_info: str = None,
    ):
        """Logs metrics for an epoch with paper-ready formatting to console, log file, and CSV."""
        t_train = train_time if train_time is not None else epoch_time
        t_val = val_time if val_time is not None else 0.0
        lr_val = learning_rate if learning_rate is not None else 0.0
        lr_str = f"{lr_val:.6f}" if learning_rate is not None else "N/A"

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
                f"{t_train:.2f}",
                f"{t_val:.2f}",
                f"{samples_per_sec:.2f}",
                lr_str,
            ])

        # Publication-grade console and file log banner
        ep_str = f"{epoch:03d}/{total_epochs:03d}" if total_epochs else f"{epoch:03d}"
        best_tag = " [NEW BEST]" if is_best else ""
        best_info = f" (Best: {best_val_acc * 100:.2f}%{best_tag})" if best_val_acc is not None else ""
        batch_str = f" | Effective Batch: {cluster_info}" if cluster_info else ""

        self.logger.info("=" * 88)
        self.logger.info(
            f" [EPOCH {ep_str} SUMMARY] Total Duration: {epoch_time:.2f}s "
            f"(Train: {t_train:.2f}s | Val: {t_val:.2f}s)"
        )
        self.logger.info("-" * 88)
        self.logger.info(
            f"   * Train Loss:     {train_loss:.4f}         | Train Top-1 Acc: {train_accuracy * 100:6.2f}%"
        )
        self.logger.info(
            f"   * Val Loss:       {val_loss:.4f}         | Val Top-1 Acc:   {val_accuracy * 100:6.2f}%{best_info}"
        )
        self.logger.info(
            f"   * Cluster Tput:   {samples_per_sec:6.1f} samples/s{batch_str}"
        )
        if learning_rate is not None:
            self.logger.info(f"   * Learning Rate:  {lr_val:.3e}")
        self.logger.info("=" * 88)

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)
