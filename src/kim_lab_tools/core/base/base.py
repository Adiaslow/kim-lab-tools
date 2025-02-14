"""Base classes and interfaces for Kim Lab Tools.

This module contains the core abstractions and base classes that implement
the fundamental functionality of the package.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol


class DataProcessor(Protocol):
    """Protocol defining the interface for data processing components."""

    def process(self, data: Any) -> Any:
        """Process the input data.

        Args:
            data: The input data to process.

        Returns:
            The processed data.
        """
        ...


class BaseProcessor(ABC):
    """Abstract base class for data processors.

    This class provides a template for creating data processors that follow
    the Single Responsibility Principle and are easily extensible.
    """

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate the input data.

        Args:
            data: The input data to validate.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        pass

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process the input data.

        Args:
            data: The input data to process.

        Returns:
            The processed data.
        """
        pass
