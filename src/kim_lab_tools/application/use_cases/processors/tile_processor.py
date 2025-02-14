"""Image tiling processor implementation."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError
from ..models.section_image import SectionImage


class TileProcessor(BaseProcessor):
    """Processor for image tiling operations."""

    def __init__(self, tile_size: int = 303):
        """Initialize processor.

        Args:
            tile_size: Size of output tiles.
        """
        self.tile_size = tile_size

    def make_tiles(self, section: SectionImage) -> List[np.ndarray]:
        """Create tiles from section image.

        Args:
            section: Input section image.

        Returns:
            List of tile arrays.
        """
        tiles = []
        height, width = section.shape

        for i in range(0, height, self.tile_size):
            for j in range(0, width, self.tile_size):
                tile = section.image[i : i + self.tile_size, j : j + self.tile_size]
                tiles.append(tile)

        return tiles

    def save_tiles(
        self,
        section: SectionImage,
        output_dir: Path,
        prefix: Optional[str] = None,
    ) -> None:
        """Save tiles to directory.

        Args:
            section: Input section image.
            output_dir: Output directory.
            prefix: Optional filename prefix.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        height, width = section.shape
        row_count = height // self.tile_size + (1 if height % self.tile_size else 0)
        col_count = width // self.tile_size + (1 if width % self.tile_size else 0)

        tiles = self.make_tiles(section)
        prefix = prefix or section.stem

        for i in range(row_count):
            for j in range(col_count):
                tile_index = i * col_count + j
                if tile_index < len(tiles):
                    tile_path = output_dir / f"{prefix}_tile_{i}_{j}.tif"
                    cv2.imwrite(str(tile_path), tiles[tile_index])

    def reconstruct_from_tiles(
        self,
        input_dir: Path,
        prefix: str,
    ) -> np.ndarray:
        """Reconstruct image from tiles.

        Args:
            input_dir: Directory containing tiles.
            prefix: Filename prefix for tiles.

        Returns:
            Reconstructed image array.

        Raises:
            ProcessingError: If reconstruction fails.
        """
        try:
            # Get sorted tile paths
            tile_paths = sorted(
                Path(input_dir).glob(f"{prefix}_tile_*.tif"),
                key=lambda x: tuple(map(int, x.stem.split("_")[-2:])),
            )

            if not tile_paths:
                raise ValueError(f"No tiles found with prefix {prefix}")

            # Get tile dimensions from first tile
            first_tile = cv2.imread(str(tile_paths[0]), cv2.IMREAD_GRAYSCALE)
            tile_height, tile_width = first_tile.shape

            # Get grid dimensions
            max_row = max_col = 0
            for path in tile_paths:
                row, col = map(int, path.stem.split("_")[-2:])
                max_row = max(max_row, row)
                max_col = max(max_col, col)

            # Initialize output array
            height = (max_row + 1) * tile_height
            width = (max_col + 1) * tile_width
            reconstructed = np.zeros((height, width), dtype=np.uint8)

            # Place tiles
            for path in tile_paths:
                row, col = map(int, path.stem.split("_")[-2:])
                tile = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

                start_row = row * tile_height
                start_col = col * tile_width
                end_row = start_row + tile.shape[0]
                end_col = start_col + tile.shape[1]

                reconstructed[start_row:end_row, start_col:end_col] = tile[
                    : min(tile.shape[0], height - start_row),
                    : min(tile.shape[1], width - start_col),
                ]

            return reconstructed

        except Exception as e:
            raise ProcessingError(f"Failed to reconstruct image: {str(e)}")
