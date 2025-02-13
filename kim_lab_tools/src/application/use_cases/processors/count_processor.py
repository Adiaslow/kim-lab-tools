"""Object counting processor implementation."""

import csv
from pathlib import Path
from typing import Dict, Any

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError


class CountProcessor(BaseProcessor):
    """Processor for counting objects in regions."""

    def validate(self, data: tuple[Path, Path]) -> bool:
        """Validate input files.

        Args:
            data: Tuple of (objects_file, structures_file).

        Returns:
            bool: True if valid.

        Raises:
            ValidationError: If files are invalid.
        """
        objects_file, _ = data
        if not objects_file.exists():
            raise ValueError(f"Objects file not found: {objects_file}")
        return True

    def process(self, data: tuple[Path, Path]) -> Dict[str, Dict[str, int]]:
        """Process object counts.

        Args:
            data: Tuple of (objects_file, structures_file).

        Returns:
            Dict mapping sections to region counts.
        """
        self.validate(data)
        objects_file, structures_file = data

        try:
            # Read objects
            objects = self._read_objects(objects_file)

            # Read structures
            with open(structures_file, "rb") as f:
                regions = pickle.load(f)

            # Count objects
            return self._count_objects(objects, regions)

        except Exception as e:
            raise ProcessingError(f"Failed to process counts: {str(e)}")

    def _read_objects(self, objects_file: Path) -> Dict[str, Dict[str, int]]:
        """Read objects from CSV file.

        Args:
            objects_file: Path to objects CSV.

        Returns:
            Dict mapping sections to region counts.
        """
        objects = {}
        with open(objects_file) as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)  # Skip header
            for row in reader:
                section = row[0]
                region = row[6]
                if section in objects:
                    objects[section][region] = objects[section].get(region, 0) + 1
                else:
                    objects[section] = {region: 1}
        return objects

    def _count_objects(
        self, objects: Dict[str, Dict[str, int]], regions: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """Count objects by region.

        Args:
            objects: Dict mapping sections to region counts.
            regions: Dict mapping region IDs to metadata.

        Returns:
            Dict mapping region IDs to counts.
        """
        sums = {}
        for section_data in objects.values():
            for region_id, count in section_data.items():
                region_id = int(region_id)
                region_info = regions[region_id]

                # Handle layered regions
                if "layer" in region_info["name"].lower():
                    parent_id = region_info["parent"]
                    if parent_id:
                        sums[parent_id] = sums.get(parent_id, 0) + count
                else:
                    sums[region_id] = sums.get(region_id, 0) + count

        return sums
