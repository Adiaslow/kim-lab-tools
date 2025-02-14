"""Detection result model implementation."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DetectionResult:
    """Container for detection results."""

    boxes: List[List[float]]  # [x1, y1, x2, y2]
    scores: List[float]
    image_dimensions: Tuple[int, int]  # (height, width)
