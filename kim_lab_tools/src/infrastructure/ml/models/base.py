"""
# src/kim_lab_tools/models/base.py

Base classes for data models in kim_lab_tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseModel(ABC):
    """Base class for all models."""

    @abstractmethod
    def validate(self) -> bool:
        """Validate model data."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert model to dictionary."""
        pass
