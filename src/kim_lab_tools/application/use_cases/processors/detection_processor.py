"""Object detection processor implementation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from skimage.exposure import equalize_adapthist
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError
from ..models.detection import DetectionResult


class DetectionProcessor(BaseProcessor):
    """Processor for object detection in images."""

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence: float = 0.85,
        tile_size: int = 640,
        area_threshold: float = 200,
        eccentricity_threshold: Optional[float] = None,
        device: str = "cuda:0",
    ):
        """Initialize the processor.

        Args:
            model_path: Path to detection model.
            confidence: Detection confidence threshold.
            tile_size: Size of image tiles for processing.
            area_threshold: Minimum object area.
            eccentricity_threshold: Maximum object eccentricity.
            device: Device to run model on.
        """
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=str(model_path),
            confidence_threshold=confidence,
            device=device,
        )
        self.tile_size = tile_size
        self.area_threshold = area_threshold
        self.eccentricity_threshold = eccentricity_threshold

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

    def process(self, image: np.ndarray) -> DetectionResult:
        """Process image for object detection.

        Args:
            image: Input image array.

        Returns:
            DetectionResult containing boxes and scores.

        Raises:
            ProcessingError: If detection fails.
        """
        self.validate(image)

        try:
            # Preprocess image
            processed = self._preprocess_image(image)

            # Run detection
            result = get_sliced_prediction(
                processed,
                self.model,
                slice_height=self.tile_size,
                slice_width=self.tile_size,
                overlap_height_ratio=0.1,
                overlap_width_ratio=0.1,
            )

            # Screen predictions
            filtered = self._screen_predictions(
                result.object_prediction_list,
                processed,
            )

            # Extract results
            boxes = [obj.bbox.to_xyxy() for obj in filtered]
            scores = [obj.score.value for obj in filtered]

            return DetectionResult(
                boxes=boxes,
                scores=scores,
                image_dimensions=image.shape[:2],
            )

        except Exception as e:
            raise ProcessingError(f"Detection failed: {str(e)}")

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection."""
        # Handle different dtypes
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        elif image.dtype in (np.float32, np.float64):
            image = (image * 255).astype(np.uint8)

        # Enhance contrast
        image = equalize_adapthist(image, clip_limit=0.01)
        image = (image * 255).astype(np.uint8)

        # Convert to BGR if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return image

    def _screen_predictions(self, predictions: List, image: np.ndarray) -> List:
        """Screen predictions based on area and eccentricity."""
        # Filter by area
        first_pass = [
            obj
            for obj in predictions
            if self._get_area(obj.bbox.to_xyxy()) > self.area_threshold
        ]

        if len(first_pass) < 3:
            return first_pass

        # Filter outlier areas
        areas = [self._get_area(obj.bbox.to_xyxy()) for obj in first_pass]
        avg_area = np.mean(areas)
        std_area = np.std(areas)
        second_pass = [
            obj
            for obj in first_pass
            if self._get_area(obj.bbox.to_xyxy()) < avg_area + 2 * std_area
        ]

        # Filter by eccentricity if needed
        if self.eccentricity_threshold is not None:
            second_pass = [
                obj
                for obj in second_pass
                if not self._check_eccentricity(obj.bbox.to_xyxy(), image)
            ]

        return second_pass

    def _get_area(self, box: List[float]) -> float:
        """Calculate box area."""
        return (box[2] - box[0]) * (box[3] - box[1])

    def _check_eccentricity(self, box: List[float], image: np.ndarray) -> bool:
        """Check object eccentricity."""
        try:
            # Extract and process cell region
            box = [int(b) for b in box]
            cell = image[box[1] - 5 : box[3] + 5, box[0] - 5 : box[2] + 5]
            if len(cell.shape) > 2:
                cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

            # Create and analyze mask
            mask = cell > threshold_otsu(cell)
            regions = regionprops(label(mask))
            if not regions:
                return False

            largest = max(regions, key=lambda r: r.area)
            return largest.eccentricity > self.eccentricity_threshold

        except Exception:
            return True
