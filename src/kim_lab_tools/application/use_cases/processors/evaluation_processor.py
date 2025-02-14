"""Axon detection evaluation processor implementation."""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.ndimage import binary_dilation
from skimage.morphology import remove_small_objects

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    dice_coefficient: float
    detection_overlay: np.ndarray
    edges: np.ndarray


class EvaluationProcessor(BaseProcessor):
    """Processor for axon detection evaluation."""

    def __init__(
        self,
        margin: int = 5,
        min_size: int = 50,
        gaussian_params: tuple = ((7, 7), 0.5, 20),
    ):
        """Initialize processor.

        Args:
            margin: Margin of error in pixels.
            min_size: Minimum object size.
            gaussian_params: Tuple of (kernel_size, sigma1, sigma2).
        """
        self.margin = margin
        self.min_size = min_size
        self.kernel_size, self.sigma1, self.sigma2 = gaussian_params

    def preprocess_label(self, label: np.ndarray) -> np.ndarray:
        """Preprocess label image.

        Args:
            label: Input label image.

        Returns:
            Preprocessed binary label.
        """
        # Invert and binarize
        inverted = cv2.bitwise_not(label)
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using DoG.

        Args:
            image: Input grayscale image.

        Returns:
            Binary edge map.
        """
        # Apply Gaussian blur
        gauss_a = cv2.GaussianBlur(image, self.kernel_size, self.sigma1)
        gauss_b = cv2.GaussianBlur(image, self.kernel_size, self.sigma2)

        # Difference of Gaussians
        dog = gauss_b - gauss_a
        dog = cv2.normalize(dog, None, 0, 1, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold and clean up
        _, edges = cv2.threshold(dog, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = binary_dilation(edges, structure=np.ones((2, 2)), iterations=1)
        edges = remove_small_objects(edges, min_size=self.min_size)

        return edges.astype(np.uint8) * 255

    def create_overlay(
        self,
        label: np.ndarray,
        edges: np.ndarray,
        dilated_label: np.ndarray,
    ) -> np.ndarray:
        """Create visualization overlay.

        Args:
            label: Original label image.
            edges: Detected edges.
            dilated_label: Dilated label for margin.

        Returns:
            Color overlay image.
        """
        # Create color base image
        base = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
        base[:, :, 0] = dilated_label * 255  # Blue channel
        base[:, :, 1] = 0  # Green channel
        base[:, :, 2] = 0  # Red channel

        # Create red overlay for edges
        overlay = np.zeros_like(base)
        overlay[:, :, 2] = edges

        # Combine images
        return cv2.addWeighted(base, 0.5, overlay, 1.0, 0)

    def compute_dice(
        self,
        label_binary: np.ndarray,
        edges_binary: np.ndarray,
    ) -> float:
        """Compute Dice coefficient.

        Args:
            label_binary: Binary label image.
            edges_binary: Binary edge image.

        Returns:
            Dice coefficient.
        """
        intersection = np.sum(label_binary * edges_binary)
        union = np.sum(label_binary) + np.sum(edges_binary) - intersection
        return 2 * intersection / union if union > 0 else 0.0

    def evaluate(
        self,
        image: np.ndarray,
        label: np.ndarray,
    ) -> EvaluationResult:
        """Evaluate detection against label.

        Args:
            image: Input grayscale image.
            label: Ground truth label image.

        Returns:
            Evaluation results.

        Raises:
            ProcessingError: If evaluation fails.
        """
        try:
            # Preprocess label
            label_binary = self.preprocess_label(label) // 255

            # Detect edges
            edges = self.detect_edges(image)
            edges_binary = edges // 255

            # Create dilated label for margin
            kernel = np.ones((self.margin, self.margin), np.uint8)
            label_dilated = cv2.dilate(label_binary, kernel)

            # Create visualization
            overlay = self.create_overlay(label, edges, label_dilated)

            # Compute metrics
            dice = self.compute_dice(label_dilated, edges_binary)

            return EvaluationResult(dice, overlay, edges)

        except Exception as e:
            raise ProcessingError(f"Evaluation failed: {str(e)}")

    def evaluate_file(
        self,
        image_path: Path,
        label_path: Path,
        show_results: bool = True,
    ) -> EvaluationResult:
        """Evaluate detection on image files.

        Args:
            image_path: Path to input image.
            label_path: Path to label image.
            show_results: Whether to display results.

        Returns:
            Evaluation results.
        """
        # Load images
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)

        if image is None or label is None:
            raise ValueError("Failed to load images")

        # Run evaluation
        result = self.evaluate(image, label)

        # Display results if requested
        if show_results:
            cv2.imshow("Edges", result.edges)
            cv2.imshow("Detection Overlay", result.detection_overlay)
            print(
                f"Dice Coefficient with {self.margin}px margin: "
                f"{result.dice_coefficient:.4f}"
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result
