"""Atlas region model implementation."""

from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np


@dataclass
class AtlasRegion:
    """Container for atlas region data."""

    id: int
    name: str
    acronym: str
    color: np.ndarray
    id_path: str

    @classmethod
    def from_class_map(
        cls, region_id: int, region_data: Dict[str, Any]
    ) -> "AtlasRegion":
        """Create region from class map data.

        Args:
            region_id: Region identifier.
            region_data: Region metadata.

        Returns:
            AtlasRegion instance.
        """
        return cls(
            id=region_id,
            name=region_data["name"],
            acronym=region_data["acronym"],
            color=np.random.randint(0, 255, size=3),
            id_path=region_data.get("id_path", ""),
        )

    @property
    def parent_id(self) -> int:
        """Get parent region ID from path."""
        if not self.id_path:
            return self.id
        return int(self.id_path.split("/")[-3])

    def is_layer(self) -> bool:
        """Check if region is a layer."""
        return "layer" in self.name.lower()
