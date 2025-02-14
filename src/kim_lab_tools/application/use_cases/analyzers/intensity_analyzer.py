"""Intensity analysis implementation."""

from typing import Dict, List, Any
import numpy as np

from ..core.base import BaseProcessor
from ..core.exceptions import ValidationError


class IntensityAnalyzer(BaseProcessor):
    """Analyzer for region intensities."""

    def __init__(
        self, structure_map: Dict[int, Dict[str, Any]], whole_brain: bool = False
    ):
        """Initialize the analyzer.

        Args:
            structure_map: Dictionary mapping structure IDs to their metadata.
            whole_brain: Whether to process whole brain or just left half.
        """
        self.structure_map = structure_map
        self.whole_brain = whole_brain
        self.required_regions = [
            "VISa",
            "VISal",
            "VISam",
            "VISp",
            "VISl",
            "VISli",
            "VISpl",
            "VISpm",
            "VISpor",
            "VISrl",
            "RSPagl",
            "RSPd",
            "RSPv",
        ]

    def validate(self, data: tuple[np.ndarray, np.ndarray]) -> bool:
        """Validate intensity and annotation data.

        Args:
            data: Tuple of (intensity_image, annotation_data).

        Returns:
            bool: True if data is valid.
        """
        intensity, annotation = data
        if intensity.shape != annotation.shape:
            raise ValidationError("Intensity and annotation shapes don't match")
        return True

    def process(self, data: tuple[np.ndarray, np.ndarray]) -> Dict[int, Dict]:
        """Analyze intensity data for regions.

        Args:
            data: Tuple of (intensity_image, annotation_data).

        Returns:
            Dict mapping region IDs to their intensity data.
        """
        intensity, annotation = data
        self.validate((intensity, annotation))

        height, width = intensity.shape
        required_ids = [
            atlas_id
            for atlas_id, data in self.structure_map.items()
            if data["acronym"] in self.required_regions
        ]

        intensities = {required_id: {} for required_id in required_ids}
        children_ids = self._get_children_ids(required_ids)

        return self._collect_intensities(children_ids, annotation, intensity, width)

    def _get_children_ids(self, required_ids: List[int]) -> Dict[int, List[int]]:
        """Get child IDs for each required region."""
        children_ids = {required_id: [] for required_id in required_ids}
        for required_id in required_ids:
            for atlas_id, data in self.structure_map.items():
                if required_id in [
                    int(sub_id) for sub_id in data["id_path"].split("/")
                ]:
                    children_ids[required_id].append(atlas_id)
        return children_ids

    def _collect_intensities(
        self,
        children_ids: Dict[int, List[int]],
        annotation: np.ndarray,
        intensity: np.ndarray,
        width: int,
    ) -> Dict[int, Dict]:
        """Collect intensity values for each region."""
        intensities = {parent_id: {} for parent_id in children_ids.keys()}

        for parent_id, children in children_ids.items():
            for child_id in children:
                verts = np.where(annotation == child_id)
                if verts[0].size == 0:
                    continue

                for point in zip(*verts):
                    if not self.whole_brain or point[1] < width // 2:
                        intensities[parent_id][point] = intensity[point]

        return intensities
