"""Concrete implementations of data processors."""

from typing import Any
import numpy as np

from .base import BaseProcessor
from .exceptions import ValidationError, ProcessingError


class NumericDataProcessor(BaseProcessor):
    """Processor for numeric data arrays."""

    def validate(self, data: Any) -> bool:
        """Validate that the input is a numeric array.

        Args:
            data: Input data to validate.

        Returns:
            bool: True if the data is valid.

        Raises:
            ValidationError: If the data is not numeric or is empty.
        """
        try:
            arr = np.asarray(data)
            if arr.size == 0:
                raise ValidationError("Input array is empty")
            if not np.issubdtype(arr.dtype, np.number):
                raise ValidationError("Input array must be numeric")
            return True
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid input data: {str(e)}")

    def process(self, data: Any) -> np.ndarray:
        """Process numeric data.

        Args:
            data: Input numeric data.

        Returns:
            np.ndarray: Processed data.

        Raises:
            ProcessingError: If processing fails.
        """
        self.validate(data)
        try:
            return np.asarray(data)
        except Exception as e:
            raise ProcessingError(f"Failed to process data: {str(e)}")
