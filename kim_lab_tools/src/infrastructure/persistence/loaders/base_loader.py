"""
# src/kim_lab_tools/loaders/base_loader.py

Base classes for data loaders.
"""

from ..core.interfaces import DataLoader
from typing import Any
from pathlib import Path


class BaseLoader(DataLoader):
    """Base class for all data loaders."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def load(self, path: str) -> Any:
        """Load data from path."""
        full_path = self.root_dir / path
        self._validate_path(full_path)
        return self._load_implementation(full_path)

    def _validate_path(self, path: Path) -> None:
        """Validate file path."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    def _load_implementation(self, path: Path) -> Any:
        """Implementation of loading logic."""
        raise NotImplementedError
