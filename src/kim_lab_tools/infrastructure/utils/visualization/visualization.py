"""Visualization utilities."""

import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import List, Union, Tuple

from ..models.detection import DetectionResult


def create_prediction_overlay(
    image_shape: Tuple[int, int, int],
    predictions: List[DetectionResult],
    dot_color: Tuple[int, int, int] = (0, 0, 255),
    dot_radius: int = 8,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create overlay image with prediction dots.

    Args:
        image_shape: Shape of original image (height, width, channels).
        predictions: List of detection results.
        dot_color: Color for prediction dots in BGR.
        dot_radius: Radius of prediction dots.
        background_color: Background color in BGR.

    Returns:
        Overlay image array.
    """
    height, width, channels = image_shape
    overlay = np.zeros((height, width, channels))
    overlay[:, :, :] = background_color

    for pred in predictions:
        for box in pred.boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            cv2.circle(overlay, (center_x, center_y), dot_radius, dot_color, -1)

    return overlay.astype(np.uint8)


def save_prediction_overlays(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    prediction_suffix: str = "Predictions",
) -> None:
    """Save prediction overlays for all images.

    Args:
        input_dir: Directory containing original images.
        output_dir: Directory for output overlays.
        prediction_suffix: Suffix for prediction files.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for img_path in input_dir.glob("*.*"):
        if not img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            continue

        # Load image and predictions
        img = cv2.imread(str(img_path))
        pred_path = output_dir / f"{prediction_suffix}_{img_path.stem}.pkl"

        try:
            with open(pred_path, "rb") as f:
                predictions = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError) as e:
            print(f"Failed to load predictions for {img_path.name}: {e}")
            continue

        # Create and save overlay
        overlay = create_prediction_overlay(img.shape, predictions)
        output_path = output_dir / f"{prediction_suffix}_{img_path.stem}.png"
        cv2.imwrite(str(output_path), overlay)


def draw_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    color: tuple = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: Input image.
        boxes: List of [x1, y1, x2, y2] boxes.
        color: Box color in BGR.
        thickness: Line thickness.

    Returns:
        Image with drawn boxes.
    """
    result = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    return result


def export_boxes(
    image: np.ndarray,
    boxes: List[List[float]],
    output_path: Path,
) -> None:
    """Export image with drawn boxes.

    Args:
        image: Input image.
        boxes: List of [x1, y1, x2, y2] boxes.
        output_path: Path to save result.
    """
    result = draw_boxes(image, boxes)
    cv2.imwrite(str(output_path), result)
