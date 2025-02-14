"""Image filtering processor implementation."""

import numpy as np
import tifffile as tf
from pathlib import Path
from typing import Union, Optional

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError
from ..utils.image_processing import (
    normalize_bit_depth,
    apply_tophat,
    adjust_gamma,
)


class FilterProcessor(BaseProcessor):
    """Processor for image filtering operations."""

    def __init__(
        self,
        filter_size: int,
        gamma: float = 1.25,
        target_type: str = "uint8",
    ):
        """Initialize processor.

        Args:
            filter_size: Size of tophat filter.
            gamma: Gamma correction value.
            target_type: Target image data type.
        """
        self.filter_size = filter_size
        self.gamma = gamma
        self.target_type = target_type

    def validate(self, image: np.ndarray) -> bool:
        """Validate input image.

        Args:
            image: Input image array.

        Returns:
            bool: True if valid.

        Raises:
            ValidationError: If image is invalid.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a numpy array")
        if image.size == 0:
            raise ValueError("Input image is empty")
        return True

    def process(self, image: np.ndarray) -> np.ndarray:
        """Process image with filters.

        Args:
            image: Input image array.

        Returns:
            Processed image array.

        Raises:
            ProcessingError: If processing fails.
        """
        try:
            self.validate(image)

            # Normalize bit depth
            image = normalize_bit_depth(image, self.target_type)

            # Apply filters
            filtered = apply_tophat(image, self.filter_size)
            filtered = adjust_gamma(filtered, self.gamma)

            return filtered

        except Exception as e:
            raise ProcessingError(f"Failed to process image: {str(e)}")

    def process_file(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Process image file.

        Args:
            input_path: Path to input file.
            output_path: Optional path to save output.
        """
        input_path = Path(input_path)
        if output_path:
            output_path = Path(output_path)
        else:
            output_path = input_path.parent / f"filtered_{input_path.name}"

        try:
            # Load and process image
            image = tf.imread(str(input_path))
            processed = self.process(image)

            # Save result
            tf.imwrite(str(output_path), processed)

        except Exception as e:
            print(f"Failed to process {input_path.name}: {str(e)}")
