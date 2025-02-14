"""
Core interfaces and abstract base classes for kim_lab_tools.

This module defines the fundamental interfaces that different components
of the system must implement, following the dependency inversion principle.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class ImageProcessor(Protocol):
    """Protocol defining image processing operations."""

    @abstractmethod
    def process(self, image: Any) -> Any:
        """Process an input image."""
        pass


class DataLoader(Protocol):
    """Protocol defining data loading operations."""

    @abstractmethod
    def load(self, path: str) -> Any:
        """Load data from the specified path."""
        pass


class Reconstructor(Protocol):
    """Protocol defining reconstruction operations."""

    @abstractmethod
    def reconstruct(self, data: Any) -> Any:
        """Reconstruct data into desired format."""
        pass
