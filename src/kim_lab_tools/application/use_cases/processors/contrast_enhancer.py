"""Contrast enhancement implementation."""

import numpy as np
import cv2

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


class ContrastEnhancer(BaseProcessor):
    """Enhancer for image contrast."""

    def __init__(self, saturation_level: float = 0.05):
        """Initialize the enhancer.

        Args:
            saturation_level: Percentage of pixels to saturate.
        """
        self.saturation_level = saturation_level

    def validate(self, data: np.ndarray) -> bool:
        """Validate the input image.

        Args:
            data: Input image array.

        Returns:
            bool: True if valid.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if not data.size:
            raise ValueError("Input array is empty")
        return True

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast by saturating pixels.

        Args:
            image: Input image array.

        Returns:
            np.ndarray: Enhanced image.
        """
        saturation_point = self.saturation_level / 100
        flat_image = image.ravel()

        low_saturation_value = np.percentile(flat_image, saturation_point)
        high_saturation_value = np.percentile(flat_image, 100 - saturation_point)

        clipped_image = np.clip(flat_image, low_saturation_value, high_saturation_value)

        if np.issubdtype(image.dtype, np.integer):
            dtype_min, dtype_max = np.iinfo(image.dtype).min, np.iinfo(image.dtype).max
        else:
            dtype_min, dtype_max = np.finfo(image.dtype).min, np.finfo(image.dtype).max

        rescaled_image = np.interp(
            clipped_image,
            (clipped_image.min(), clipped_image.max()),
            (dtype_min, dtype_max),
        )

        return rescaled_image.reshape(image.shape).astype(image.dtype)

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process the image.

        Args:
            data: Input image array.

        Returns:
            np.ndarray: Enhanced image.

        Raises:
            ProcessingError: If enhancement fails.
        """
        self.validate(data)
        try:
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(data)
            # Apply contrast enhancement
            return self.enhance_contrast(enhanced)
        except Exception as e:
            raise ProcessingError(f"Failed to enhance contrast: {str(e)}")
