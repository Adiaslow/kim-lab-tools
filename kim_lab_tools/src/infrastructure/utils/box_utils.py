"""Box operation utilities."""

import numpy as np
from typing import List, Tuple, Union

Box = List[Union[int, float]]  # [xmin, ymin, xmax, ymax]


def compute_iou(box_a: Box, box_b: Box) -> float:
    """Compute Intersection over Union between two boxes.

    Args:
        box_a: First box coordinates [xmin, ymin, xmax, ymax].
        box_b: Second box coordinates [xmin, ymin, xmax, ymax].

    Returns:
        IoU value between 0 and 1.
    """
    # Intersection coordinates
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Intersection area
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    # Box areas
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # IoU
    return inter_area / float(box_a_area + box_b_area - inter_area)


def compute_overlaps(boxes1: List[Box], boxes2: List[Box]) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes.

    Args:
        boxes1: First list of boxes.
        boxes2: Second list of boxes.

    Returns:
        Matrix of IoU values.
    """
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            overlaps[i, j] = compute_iou(box1, box2)
    return overlaps


def compute_colocalization(
    boxes1: List[Box], boxes2: List[Box], threshold: float = 0.5
) -> float:
    """Compute percentage of colocalized boxes.

    Args:
        boxes1: First list of boxes.
        boxes2: Second list of boxes.
        threshold: IoU threshold for colocalization.

    Returns:
        Percentage of colocalized boxes.
    """
    if not boxes1 or not boxes2:
        return 0.0

    overlaps = compute_overlaps(boxes1, boxes2)
    max_overlaps = np.max(overlaps, axis=1)
    colocalized_count = np.sum(max_overlaps > threshold)

    return (colocalized_count / len(boxes1)) * 100
