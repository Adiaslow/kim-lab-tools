"""Synthetic experiment generation implementation."""

from pathlib import Path
import cv2
import numpy as np
from uuid import uuid4
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .slice_generator import SliceGenerator
from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


class ExperimentGenerator(BaseProcessor):
    """Generator for synthetic experiments."""

    def __init__(
        self,
        output_path: Path,
        num_samples: int,
        output_size: tuple[int, int] = (224, 224),
        max_workers: int = 128,
    ):
        """Initialize the experiment generator.

        Args:
            output_path: Path to save experiment data.
            num_samples: Number of samples to generate.
            output_size: Size of output images.
            max_workers: Maximum number of concurrent workers.
        """
        self.output_path = Path(output_path)
        self.num_samples = num_samples
        self.output_size = output_size
        self.max_workers = max_workers
        self.slice_generator = SliceGenerator()

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

    def _generate_sample(
        self, i: int, volume: np.ndarray, experiment_path: Path, metadata_file: Path
    ) -> None:
        """Generate a single sample."""
        params = {
            "z_position": np.random.randint(200, 1200),
            "x_angle": np.random.uniform(-15, 15),
            "y_angle": np.random.uniform(-15, 15),
            "output_size": self.output_size,
            "half_brain": np.random.rand() > 0.5,
        }

        result = self.slice_generator.process((volume, params))
        sample = result["slice"]
        metadata = result["metadata"]

        # Save sample
        sample_name = f"S_{uuid4()}.png"
        cv2.imwrite(str(experiment_path / sample_name), sample)

        # Update metadata
        with open(metadata_file, "a") as f:
            f.write(
                f"{sample_name},{metadata['x_angle']},{metadata['y_angle']},"
                f"{metadata['z_position']}\n"
            )

    def process(self, data: np.ndarray) -> None:
        """Generate the synthetic experiment.

        Args:
            data: 3D volume array to generate samples from.
        """
        self.validate(data)

        try:
            # Setup output directories
            self.output_path.mkdir(exist_ok=True)
            experiment_path = self.output_path
            experiment_path.mkdir(exist_ok=True)

            # Initialize metadata file
            metadata_file = experiment_path / "metadata.csv"
            with open(metadata_file, "w") as f:
                f.write("filename,x_angle,y_angle,z_position\n")

            # Generate samples
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._generate_sample,
                        i,
                        data,
                        experiment_path,
                        metadata_file,
                    )
                    for i in range(self.num_samples)
                ]
                for i, future in enumerate(as_completed(futures)):
                    future.result()
                    if i % 100 == 0:
                        print(f"Completed {i}/{self.num_samples} samples", end="\r")

        except Exception as e:
            raise ProcessingError(f"Failed to generate experiment: {str(e)}")
