"""Image processing utility functions."""

import numpy as np
import cv2
from typing import Union


def adjust_gamma(
    image: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply gamma correction to image.

    Args:
        image: Input image array.
        gamma: Gamma correction value.

    Returns:
        Gamma corrected image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
    ).astype(np.uint8)
    return cv2.LUT(image, table)


def normalize_bit_depth(
    image: np.ndarray,
    target_type: str = "uint8",
) -> np.ndarray:
    """Normalize image bit depth.

    Args:
        image: Input image array.
        target_type: Target data type.

    Returns:
        Normalized image.
    """
    if image.dtype == "uint16" and target_type == "uint8":
        return (image / 256).astype(target_type)
    return image.astype(target_type)


def apply_tophat(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Apply tophat filter to image.

    Args:
        image: Input image array.
        kernel_size: Size of filter kernel.

    Returns:
        Filtered image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
