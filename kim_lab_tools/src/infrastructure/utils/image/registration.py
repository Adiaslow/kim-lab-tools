"""Registration utility functions."""

import SimpleITK as sitk
import numpy as np
from skimage.filters import sobel
import cv2
from typing import Tuple


def match_histograms(fixed: sitk.Image, moving: sitk.Image) -> sitk.Image:
    """Match moving image histogram to fixed image.

    Args:
        fixed: Reference image.
        moving: Image to match.

    Returns:
        Histogram matched image.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving, fixed)


def preprocess_image(image: sitk.Image) -> sitk.Image:
    """Preprocess image to enhance features.

    Args:
        image: Input image.

    Returns:
        Preprocessed image.
    """
    array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkUInt8))
    blurred = cv2.GaussianBlur(array, (5, 5), 0)
    edges = sobel(blurred)
    edges = (edges - np.min(edges)) / (np.max(edges) - np.min(edges))
    return sitk.GetImageFromArray(edges.astype(np.float32))


def multimodal_registration(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """Perform multimodal image registration.

    Args:
        fixed: Reference image.
        moving: Image to register.

    Returns:
        Composite transform.
    """
    fixed = preprocess_image(fixed)
    moving = preprocess_image(moving)

    # Affine registration
    initial_tx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation()
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.01,
        numberOfIterations=300,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 0])
    reg.SetInitialTransform(initial_tx)
    reg.SetInterpolator(sitk.sitkLinear)

    affine_tx = reg.Execute(fixed, moving)

    # B-spline registration
    resampled = sitk.Resample(
        moving, fixed, affine_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )

    mesh_size = [6] * fixed.GetDimension()
    bspline_tx = sitk.BSplineTransformInitializer(fixed, mesh_size)

    reg.SetMetricAsANTSNeighborhoodCorrelation(11)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(bspline_tx, inPlace=False)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.001,
        numberOfIterations=300,
        convergenceMinimumValue=1e-10,
        convergenceWindowSize=20,
    )

    bspline_tx = reg.Execute(fixed, resampled)

    # Combine transforms
    composite = sitk.CompositeTransform(affine_tx)
    composite.AddTransform(bspline_tx)
    return composite


def apply_transform(
    moving: sitk.Image,
    fixed: sitk.Image,
    transform: sitk.Transform,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Apply transform to moving image.

    Args:
        moving: Image to transform.
        fixed: Reference image.
        transform: Transform to apply.
        interpolator: Interpolation method.
        default_value: Value for regions outside image.

    Returns:
        Transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(transform)
    return resampler.Execute(moving)
