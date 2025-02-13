"""ROI analysis processor implementation."""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from skimage.filters import threshold_triangle, sobel, gaussian
from skimage.morphology import remove_small_objects

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError
from ..models.roi_analysis import (
    ROIAnalysisResult,
    AnalysisParameters,
    ExperimentGroup,
)


class ROIProcessor(BaseProcessor):
    """Processor for ROI analysis."""

    def __init__(self, params: Optional[AnalysisParameters] = None):
        """Initialize processor.

        Args:
            params: Analysis parameters.
        """
        self.params = params or AnalysisParameters()

    def process_image(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Process ROI image.

        Args:
            image: Input image array.
            mask: ROI mask array.

        Returns:
            Processed binary mask.
        """
        # Adjust contrast and brightness
        image = np.clip(
            self.params.contrast * image + self.params.brightness,
            0,
            255,
        ).astype(np.uint8)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

        # Apply filters
        smoothed = gaussian(image, sigma=self.params.sigma)
        edges = sobel(smoothed)
        edges = (edges - edges.min()) / (edges.max() - edges.min())
        binary = edges > threshold_triangle(edges)
        binary = remove_small_objects(binary)

        # Correct edges using mask
        outside_points = np.argwhere(mask == 0)
        binary = self._correct_edges(outside_points, binary)

        return binary

    def _correct_edges(
        self,
        outside_points: np.ndarray,
        binary: np.ndarray,
        max_distance: int = 20,
    ) -> np.ndarray:
        """Correct binary mask edges.

        Args:
            outside_points: Points outside ROI.
            binary: Binary mask to correct.
            max_distance: Maximum correction distance.

        Returns:
            Corrected binary mask.
        """
        for point in outside_points:
            y, x = point
            if y >= binary.shape[0] or x >= binary.shape[1]:
                continue
            binary[y, x] = 0

        return binary

    def calculate_h2b_distribution(
        self,
        centers: List[Tuple[int, int]],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """Calculate H2B distribution.

        Args:
            centers: List of H2B center coordinates.
            shape: Shape of output array.

        Returns:
            H2B distribution array.
        """
        distribution = np.zeros(shape[0])
        for x, y in centers:
            if 0 <= y < shape[0]:
                distribution[y] += 1
        return distribution

    def process_roi(
        self,
        roi_path: Path,
        predictions: Optional[Dict] = None,
    ) -> Optional[ROIAnalysisResult]:
        """Process single ROI.

        Args:
            roi_path: Path to ROI file.
            predictions: Optional detection predictions.

        Returns:
            Analysis result or None if failed.
        """
        try:
            # Load and process ROI
            roi = self.load_roi(roi_path)
            if roi is None:
                return None

            # Process image
            binary = self.process_image(roi.image, roi.mask)

            # Calculate H2B distribution if predictions available
            h2b_dist = None
            if predictions:
                pred_key = f"{roi.animal_name}_{roi.section_number}"
                if pred_key in predictions:
                    centers = self._get_centers(predictions[pred_key].boxes)
                    h2b_dist = self.calculate_h2b_distribution(centers, binary.shape)

            return ROIAnalysisResult(
                name=roi.area_name,
                filename=roi_path,
                mask=binary,
                area=np.sum(binary),
                h2b_distribution=h2b_dist,
            )

        except Exception as e:
            raise ProcessingError(f"Failed to process ROI: {str(e)}")

    def _get_centers(self, boxes: np.ndarray) -> List[Tuple[int, int]]:
        """Get center points from boxes.

        Args:
            boxes: Array of bounding boxes.

        Returns:
            List of center coordinates.
        """
        centers = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        return centers
