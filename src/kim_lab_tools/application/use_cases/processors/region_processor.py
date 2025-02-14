"""Region processing implementation."""

from typing import Dict, Tuple
import numpy as np

from ..core.base import BaseProcessor
from ..core.exceptions import ValidationError, ProcessingError


class RegionProcessor(BaseProcessor):
    """Processor for region reconstruction from intensity data."""

    def validate(self, data: Dict[Tuple[int, int], float]) -> bool:
        """Validate the intensity data dictionary.

        Args:
            data: Dictionary with (x,y) coordinate tuples as keys and intensity values.

        Returns:
            bool: True if data is valid.

        Raises:
            ValidationError: If data is empty or malformed.
        """
        if not isinstance(data, dict):
            raise ValidationError("Input must be a dictionary")
        if not data:
            raise ValidationError("Input dictionary is empty")

        # Validate structure of first item
        first_key = next(iter(data.keys()))
        if not (isinstance(first_key, tuple) and len(first_key) == 2):
            raise ValidationError("Keys must be (x,y) coordinate tuples")

        return True

    def process(self, data: Dict[Tuple[int, int], float]) -> np.ndarray:
        """Process region intensity data into a 2D array.

        Args:
            data: Dictionary with (x,y) coordinate tuples as keys and intensity values.

        Returns:
            np.ndarray: 2D array of reconstructed region.

        Raises:
            ProcessingError: If processing fails.
        """
        self.validate(data)
        try:
            max_x = max(point[0] for point in data.keys())
            max_y = max(point[1] for point in data.keys())

            reconstructed = np.zeros((max_x + 1, max_y + 1))
            for point, intensity in data.items():
                reconstructed[point] = intensity

            return reconstructed
        except Exception as e:
            raise ProcessingError(f"Failed to process region: {str(e)}")
