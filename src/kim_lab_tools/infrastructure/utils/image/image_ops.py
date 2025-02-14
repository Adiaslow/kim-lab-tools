"""Image operation utilities."""

import SimpleITK as sitk
import numpy as np
from typing import Tuple


def resize_nearest(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize image using nearest neighbor interpolation.

    Args:
        image: Input image array.
        size: Target size (height, width).

    Returns:
        Resized image array.
    """
    sitk_image = sitk.GetImageFromArray(image)
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()

    new_spacing = [
        float(orig_space) * float(orig_size) / float(new_dim)
        for orig_space, orig_size, new_dim in zip(original_spacing, original_size, size)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputPixelType(sitk_image.GetPixelIDValue())
    resampler.SetSize(size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())

    resized = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(resized)


def resize_to_width(image: sitk.Image, target_width: int) -> sitk.Image:
    """Resize image to target width while maintaining aspect ratio.

    Args:
        image: Input SimpleITK image.
        target_width: Target width.

    Returns:
        Resized SimpleITK image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    aspect_ratio = original_size[1] / original_size[0]
    new_height = int(target_width * aspect_ratio)

    new_spacing = [
        original_size[0] / target_width * original_spacing[0],
        original_size[1] / new_height * original_spacing[1],
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((target_width, new_height))
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)
