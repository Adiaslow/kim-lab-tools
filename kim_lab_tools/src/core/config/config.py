"""
Configuration management utilities for kim_lab_tools.

This module provides functionality for loading and managing
configuration settings across the application.
"""

from pathlib import Path
import yaml
from typing import Dict, Any


class Config:
    """Configuration manager for kim_lab_tools."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)
