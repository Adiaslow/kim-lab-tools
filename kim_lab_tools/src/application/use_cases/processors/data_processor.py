"""Training data processor implementation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Iterator
from dataclasses import dataclass

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


@dataclass
class ImagePair:
    """Container for image-label pair."""

    image: np.ndarray
    label: np.ndarray
    name: str


class DataProcessor(BaseProcessor):
    """Processor for training data preparation."""

    def __init__(self, tile_size: int = 256):
        """Initialize processor.

        Args:
            tile_size: Size of output tiles.
        """
        self.tile_size = tile_size

    def load_image_pairs(
        self,
        image_dir: Path,
        label_dir: Path,
    ) -> List[ImagePair]:
        """Load matching image and label pairs.

        Args:
            image_dir: Directory containing images.
            label_dir: Directory containing labels.

        Returns:
            List of image-label pairs.

        Raises:
            ProcessingError: If loading fails.
        """
        try:
            # Get sorted file lists
            image_files = sorted(p for p in image_dir.glob("*.png"))
            label_files = sorted(p for p in label_dir.glob("*.png"))

            if len(image_files) != len(label_files):
                raise ValueError(
                    f"Mismatched file counts: {len(image_files)} images, "
                    f"{len(label_files)} labels"
                )

            # Load pairs
            pairs = []
            for img_path, lbl_path in zip(image_files, label_files):
                if img_path.stem != lbl_path.stem:
                    raise ValueError(
                        f"Mismatched filenames: {img_path.name}, {lbl_path.name}"
                    )

                image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                label = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)

                if image is None or label is None:
                    print(f"Failed to load {img_path.name}")
                    continue

                pairs.append(ImagePair(image, label, img_path.stem))

            return pairs

        except Exception as e:
            raise ProcessingError(f"Failed to load image pairs: {str(e)}")

    def create_tiles(self, pair: ImagePair) -> Iterator[ImagePair]:
        """Create tiles from image-label pair.

        Args:
            pair: Input image-label pair.

        Yields:
            Tiles as image-label pairs.
        """
        height, width = pair.image.shape[:2]

        for i in range(0, height, self.tile_size):
            for j in range(0, width, self.tile_size):
                # Skip if tile would exceed image bounds
                if i + self.tile_size > height or j + self.tile_size > width:
                    continue

                # Extract tiles
                img_tile = pair.image[
                    i : i + self.tile_size,
                    j : j + self.tile_size,
                ]
                lbl_tile = pair.label[
                    i : i + self.tile_size,
                    j : j + self.tile_size,
                ]

                yield ImagePair(
                    img_tile,
                    lbl_tile,
                    f"{pair.name}_{i}_{j}",
                )

    def process_directory(
        self,
        image_dir: Path,
        label_dir: Path,
        output_dir: Path,
    ) -> None:
        """Process directory of images and labels.

        Args:
            image_dir: Directory containing images.
            label_dir: Directory containing labels.
            output_dir: Output directory for tiles.
        """
        # Create output directories
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Load and process pairs
        pairs = self.load_image_pairs(image_dir, label_dir)
        print(f"Found {len(pairs)} image-label pairs")

        tile_count = 0
        for pair in pairs:
            for tile in self.create_tiles(pair):
                # Save tiles
                cv2.imwrite(
                    str(images_dir / f"{tile.name}.png"),
                    tile.image,
                )
                cv2.imwrite(
                    str(labels_dir / f"{tile.name}.png"),
                    tile.label,
                )
                tile_count += 1

        print(f"Created {tile_count} tiles")
