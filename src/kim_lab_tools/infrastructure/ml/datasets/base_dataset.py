"""
# src/kim_lab_tools/datasets/base_dataset.py

Base classes for datasets.
"""

from typing import Any, Iterator
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    """Base class for all datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the size of the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Get item at index."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Iterator over dataset."""
        pass
