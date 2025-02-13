# src/application/use_cases/generators/slice_generator.py
"""3D volume slice generation implementation."""

from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from uuid import uuid4
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from kim_lab_tools.src.core.base.processors import BaseProcessor
from kim_lab_tools.src.core.exceptions import ProcessingError


class SliceGenerator(BaseProcessor):
    """Generator for 3D volume slices."""

    def validate(self, data: np.ndarray) -> bool:
        """Validate the input volume.

        Args:
            data: 3D volume array.

        Returns:
            bool: True if valid.
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 3:
            raise ValueError("Input must be a 3D numpy array")
        return True

    def slice_3d_volume(
        self, volume: np.ndarray, z_position: int, x_angle: float, y_angle: float
    ) -> np.ndarray:
        """Obtain a slice at a certain point in a 3D volume at an arbitrary angle.

        Args:
            volume: 3D numpy array.
            z_position: Position along the z-axis for the slice.
            x_angle: Angle in degrees to tilt in the x axis.
            y_angle: Angle in degrees to tilt in the y axis.

        Returns:
            2D sliced array.
        """
        x_angle_rad = np.deg2rad(x_angle)
        y_angle_rad = np.deg2rad(y_angle)

        x, y = np.meshgrid(np.arange(volume.shape[2]), np.arange(volume.shape[1]))
        z = (z_position + x * np.tan(x_angle_rad) + y * np.tan(y_angle_rad)).astype(
            np.float32
        )
        coords = np.array([z, y, x])

        return map_coordinates(volume, coords, order=0)

    def process(self, data: tuple[np.ndarray, dict]) -> dict:
        """Generate a transformed slice from the volume.

        Args:
            data: Tuple of (volume, params).
                params should contain:
                - z_position: int
                - x_angle: float
                - y_angle: float
                - output_size: tuple[int, int]
                - half_brain: bool

        Returns:
            Dict containing generated slice and metadata.
        """
        volume, params = data
        self.validate(volume)

        try:
            sample = self.slice_3d_volume(
                volume, params["z_position"], params["x_angle"], params["y_angle"]
            )

            if params.get("half_brain", False):
                removed_pixels = sample.shape[1] // 2
                sample = sample[:, : sample.shape[1] // 2]
                sample = np.pad(sample, ((0, 0), (0, removed_pixels // 2)), "constant")

            # Apply transformations
            center = (sample.shape[1] // 2, sample.shape[0] // 2)
            rotation_angle = np.random.uniform(-10, 10)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
            sample = cv2.warpAffine(
                sample, rotation_matrix, (sample.shape[1], sample.shape[0])
            )

            shear_y = np.random.rand() * 0.25 - 0.125
            shear_matrix = np.float32([[1, 0, 0], [shear_y, 1, 0]])
            sample = cv2.warpAffine(
                sample, shear_matrix, (sample.shape[1], sample.shape[0])
            )

            # Final processing
            sample = np.pad(sample, 25, "constant", constant_values=0)
            sample = cv2.resize(
                sample, params["output_size"], interpolation=cv2.INTER_LINEAR
            )

            return {
                "slice": sample,
                "metadata": {
                    "x_angle": params["x_angle"],
                    "y_angle": params["y_angle"],
                    "z_position": params["z_position"],
                    "rotation": rotation_angle,
                    "shear": shear_y,
                },
            }

        except Exception as e:
            raise ProcessingError(f"Failed to generate slice: {str(e)}")
