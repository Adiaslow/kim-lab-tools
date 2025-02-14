"""Image processing implementation."""

import numpy as np
from skimage.filters import unsharp_mask
from skimage.morphology import white_tophat, disk
import cv2

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


class ImageProcessor(BaseProcessor):
    """Processor for image enhancement and filtering."""

    def __init__(self, radius: float = 3.0, amount: float = 2.0):
        """Initialize the processor.

        Args:
            radius: Radius for unsharp mask.
            amount: Amount for unsharp mask.
        """
        self.radius = radius
        self.amount = amount

    def validate(self, data: np.ndarray) -> bool:
        """Validate the input image.

        Args:
            data: Input image array.

        Returns:
            bool: True if valid.

        Raises:
            ValidationError: If input is invalid.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not data.size:
            raise ValueError("Input array is empty")
        return True

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process the image.

        Args:
            data: Input image array.

        Returns:
            np.ndarray: Processed image.

        Raises:
            ProcessingError: If processing fails.
        """
        self.validate(data)
        try:
            original_dtype = data.dtype
            # Apply unsharp mask to enhance edges
            processed = unsharp_mask(
                data, radius=self.radius, amount=self.amount, preserve_range=True
            )
            processed = white_tophat(processed, disk(15))
            # Convert back to original dtype
            return processed.astype(original_dtype)
        except Exception as e:
            raise ProcessingError(f"Failed to process image: {str(e)}")
