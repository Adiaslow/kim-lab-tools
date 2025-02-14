"""Image loading implementation."""

from pathlib import Path
import numpy as np
import tifffile

from ..core.base import BaseProcessor
from ..core.exceptions import ValidationError, ProcessingError


class ImageLoader(BaseProcessor):
    """Loader for image files."""

    def validate(self, data: str | Path) -> bool:
        """Validate the image path.

        Args:
            data: Path to the image file.

        Returns:
            bool: True if path is valid.

        Raises:
            ValidationError: If path is invalid or file doesn't exist.
        """
        path = Path(data)
        if not path.exists():
            raise ValidationError(f"Image file not found: {path}")
        if not path.suffix.lower() in [".tif", ".tiff"]:
            raise ValidationError(f"Unsupported file format: {path.suffix}")
        return True

    def process(self, data: str | Path) -> np.ndarray:
        """Load an image file.

        Args:
            data: Path to the image file.

        Returns:
            np.ndarray: Loaded image data.

        Raises:
            ProcessingError: If image loading fails.
        """
        self.validate(data)
        try:
            return tifffile.imread(str(data))
        except Exception as e:
            raise ProcessingError(f"Failed to load image: {str(e)}")
