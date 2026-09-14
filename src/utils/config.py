"""Configuration parser and validator."""
import os
from typing import Any, Dict
import yaml


class ConfigDict(dict):
    """Dictionary that supports attribute-style access (e.g. config.training.batch_size)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'ConfigDict' object has no attribute '{key}'")


def load_config(config_path: str) -> ConfigDict:
    """Loads and validates a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        ConfigDict object containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    if config_data is None:
        raise ValueError(f"Configuration file {config_path} is empty.")

    return ConfigDict(config_data)
