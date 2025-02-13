"""Image registration processor implementation."""

import SimpleITK as sitk
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, Any
from skimage.filters import sobel

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


class RegistrationProcessor(BaseProcessor):
    """Processor for image registration."""

    def __init__(self, target_size: Tuple[int, int] = (360, 360)):
        """Initialize the processor.

        Args:
            target_size: Size to resize images to before registration.
        """
        self.target_size = target_size

    def validate(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> bool:
        """Validate input data.

        Args:
            data: Tuple of (tissue, section, label) arrays.

        Returns:
            bool: True if valid.

        Raises:
            ValidationError: If inputs are invalid.
        """
        tissue, section, label = data
        if not all(isinstance(x, np.ndarray) for x in (tissue, section, label)):
            raise ValueError("All inputs must be numpy arrays")
        if not all(x.size > 0 for x in (tissue, section, label)):
            raise ValueError("All inputs must be non-empty")
        return True

    def process(
        self, data: Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process registration.

        Args:
            data: Tuple of (tissue, section, label, structure_map).

        Returns:
            Tuple of (registered_label, registered_atlas, color_label).

        Raises:
            ProcessingError: If registration fails.
        """
        self.validate(data[:3])
        tissue, section, label, structure_map = data

        try:
            # Resize images
            tissue_resized = cv2.resize(tissue, self.target_size)
            section_resized = cv2.resize(section, self.target_size)
            label_resized = self._resize_nearest(label, self.target_size)

            # Enhance layer contrast
            section_enhanced = self._enhance_layers(
                section_resized, label_resized, structure_map
            )

            # Convert to SimpleITK
            fixed = sitk.GetImageFromArray(tissue_resized)
            moving = sitk.GetImageFromArray(section_enhanced)
            label_sitk = sitk.GetImageFromArray(label_resized)

            # Match histograms and preprocess
            fixed = self._preprocess_image(self._match_histograms(fixed, moving))
            moving = self._preprocess_image(moving)

            # Register images
            transform = self._register_multimodal(fixed, moving)

            # Apply transform
            registered_label = self._apply_transform(label_sitk, fixed, transform)
            registered_atlas = self._apply_transform(moving, fixed, transform)

            # Create color label
            color_label = self._create_color_label(registered_label, structure_map)

            # Resize back to original size
            registered_label = self._resize_nearest(registered_label, tissue.shape[:2])
            registered_atlas = cv2.resize(registered_atlas, tissue.shape[:2][::-1])
            color_label = cv2.resize(color_label, tissue.shape[:2][::-1])

            return registered_label, registered_atlas, color_label

        except Exception as e:
            raise ProcessingError(f"Registration failed: {str(e)}")

    def _preprocess_image(self, image: sitk.Image) -> sitk.Image:
        """Preprocess image to enhance features."""
        array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkUInt8))
        blurred = cv2.GaussianBlur(array, (5, 5), 0)
        edges = sobel(blurred)
        edges = (edges - np.min(edges)) / (np.max(edges) - np.min(edges))
        return sitk.GetImageFromArray(edges.astype(np.float32))

    def _match_histograms(
        self, to_match: sitk.Image, match_to: sitk.Image
    ) -> sitk.Image:
        """Match image histograms."""
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(1024)
        matcher.SetNumberOfMatchPoints(10)
        matcher.ThresholdAtMeanIntensityOn()
        return matcher.Execute(to_match, match_to)

    def _register_multimodal(
        self, fixed: sitk.Image, moving: sitk.Image
    ) -> sitk.Transform:
        """Perform multimodal registration."""
        # ... rest of registration implementation ...
