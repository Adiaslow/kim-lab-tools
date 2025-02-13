"""Atlas visualization processor implementation."""

import cv2
import numpy as np
import nrrd
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from scipy.ndimage import rotate

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError
from ..models.atlas_region import AtlasRegion


class AtlasProcessor(BaseProcessor):
    """Processor for atlas visualization."""

    LAYER_ENDINGS = ["1", "2/3", "4", "5", "6a", "6b"]

    def __init__(
        self,
        regions_to_map: List[str],
        class_map_path: Path,
        atlas_path: Path,
        rotation_angles: tuple[float, float] = (-20, -10),
    ):
        """Initialize processor.

        Args:
            regions_to_map: List of region acronyms to visualize.
            class_map_path: Path to class map pickle file.
            atlas_path: Path to atlas NRRD file.
            rotation_angles: (x,y) rotation angles in degrees.
        """
        self.regions_to_map = regions_to_map
        self.class_map_path = Path(class_map_path)
        self.atlas_path = Path(atlas_path)
        self.rotation_angles = rotation_angles
        self.class_map = self._load_class_map()
        self.annotation = self._load_atlas()

    def _load_class_map(self) -> Dict[int, AtlasRegion]:
        """Load and process class map.

        Returns:
            Dictionary mapping IDs to AtlasRegion objects.
        """
        try:
            with open(self.class_map_path, "rb") as f:
                raw_map = pickle.load(f)

            processed_map = {}
            regions_as_ids = []

            # Process regions and layers
            for region_id, region_data in raw_map.items():
                region = AtlasRegion.from_class_map(region_id, region_data)

                if region.is_layer():
                    # Check if parent matches target regions
                    acronym = region.acronym
                    for ending in self.LAYER_ENDINGS:
                        acronym = acronym.replace(ending, "")

                    if acronym.lower() in [r.lower() for r in self.regions_to_map]:
                        regions_as_ids.append(region_id)

                        # Add parent region if not present
                        if region.parent_id not in processed_map:
                            processed_map[region.parent_id] = AtlasRegion(
                                id=region.parent_id,
                                name=acronym,
                                acronym=acronym,
                                color=np.random.randint(0, 255, size=3),
                                id_path="",
                            )

                processed_map[region_id] = region

            return processed_map

        except Exception as e:
            raise ProcessingError(f"Failed to load class map: {str(e)}")

    def _load_atlas(self) -> np.ndarray:
        """Load and rotate atlas data.

        Returns:
            Rotated atlas array.
        """
        try:
            annotation, _ = nrrd.read(
                str(self.atlas_path.expanduser()), index_order="C"
            )

            # Apply rotations
            annotation = rotate(
                annotation, self.rotation_angles[0], axes=(1, 2), reshape=True, order=0
            )
            annotation = rotate(
                annotation, self.rotation_angles[1], axes=(0, 1), reshape=True, order=0
            )

            return annotation

        except Exception as e:
            raise ProcessingError(f"Failed to load atlas: {str(e)}")

    def create_dorsal_view(self) -> np.ndarray:
        """Create dorsal view visualization.

        Returns:
            Colored visualization array.
        """
        z, y, x = self.annotation.shape
        dorsal_slice_colored = np.zeros((z, x, 3), dtype=np.uint8)
        dorsal_slice_maps = {}

        # Create masks for each region
        for i in range(y):
            dorsal_slice = self.annotation[:, i, :]
            for region_id in np.unique(dorsal_slice):
                region = self.class_map.get(region_id)
                if region and region.is_layer():
                    parent_id = region.parent_id
                    if parent_id not in dorsal_slice_maps:
                        dorsal_slice_maps[parent_id] = np.zeros((z, x), dtype=np.uint8)
                    dorsal_slice_maps[parent_id][dorsal_slice == region_id] = 255

        # Remove intersections
        for mapping in sorted(dorsal_slice_maps, reverse=True):
            for other_mapping in dorsal_slice_maps:
                if mapping != other_mapping:
                    dorsal_slice_maps[mapping][
                        dorsal_slice_maps[other_mapping] == 255
                    ] = 0

        # Draw contours and labels
        for mapping, mask in dorsal_slice_maps.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dorsal_slice_colored, contours, -1, (255, 255, 255), 2)

            # Add labels
            if np.any(mask):
                c_x = int(np.mean(np.where(mask == 255)[1]))
                c_y = int(np.mean(np.where(mask == 255)[0]))
                cv2.putText(
                    dorsal_slice_colored,
                    self.class_map[mapping].acronym,
                    (c_x - 20, c_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        return dorsal_slice_colored
