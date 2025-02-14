"""
# src/kim_lab_tools/processors/base_processor.py

Base classes for image processors.
"""

from ..core.interfaces import ImageProcessor
from typing import Any


class BaseImageProcessor(ImageProcessor):
    """Base class for image processors."""

    def __init__(self):
        self.parameters = {}

    def process(self, image: Any) -> Any:
        """Process an input image."""
        self._validate_input(image)
        return self._process_implementation(image)

    def _validate_input(self, image: Any) -> None:
        """Validate input data."""
        raise NotImplementedError

    def _process_implementation(self, image: Any) -> Any:
        """Implementation of processing logic."""
        raise NotImplementedError
