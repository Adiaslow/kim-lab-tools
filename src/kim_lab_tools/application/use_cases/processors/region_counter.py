"""Region counting processor implementation."""

import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pickle

from ..core.base import BaseProcessor
from ..core.exceptions import ProcessingError
from ..utils.image import resize_image_nearest_neighbor
from ..utils.box_utils import compute_colocalization


class RegionCounter(BaseProcessor):
    """Counter for objects in brain regions."""

    def __init__(self, include_layers: bool = False):
        """Initialize the counter.

        Args:
            include_layers: Whether to count layer regions separately.
        """
        self.include_layers = include_layers

    def validate(self, data: Tuple[Path, Path, Path]) -> bool:
        """Validate input paths.

        Args:
            data: Tuple of (predictions_dir, annotations_dir, structures_file).

        Returns:
            bool: True if valid.

        Raises:
            ValidationError: If inputs are invalid.
        """
        pred_dir, anno_dir, struct_file = data
        if not all(p.exists() for p in (pred_dir, anno_dir, struct_file)):
            raise ValueError("One or more input paths do not exist")
        return True

    def process(
        self, data: Tuple[Path, Path, Path]
    ) -> Tuple[
        Dict[str, Dict[int, Dict[str, int]]], Dict[str, Dict[int, Dict[int, float]]]
    ]:
        """Process region counts and colocalization.

        Args:
            data: Tuple of (predictions_dir, annotations_dir, structures_file).

        Returns:
            Tuple of (counts_dict, colocalization_dict).

        Raises:
            ProcessingError: If processing fails.
        """
        self.validate(data)
        pred_dir, anno_dir, struct_file = data

        try:
            # Load structures
            with open(struct_file, "rb") as f:
                structures = pickle.load(f)

            # Process each section
            counts = {}
            coloc = {}

            for pred_file in sorted(pred_dir.glob("*.pkl")):
                section_name = pred_file.stem
                anno_file = anno_dir / pred_file.name

                section_counts, section_coloc = self._process_section(
                    pred_file, anno_file, structures
                )

                counts[section_name] = section_counts
                coloc[section_name] = section_coloc

            return counts, coloc

        except Exception as e:
            raise ProcessingError(f"Failed to process counts: {str(e)}")

    def _process_section(
        self, pred_file: Path, anno_file: Path, structures: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Dict[int, float]]]:
        """Process a single section.

        Args:
            pred_file: Path to predictions file.
            anno_file: Path to annotations file.
            structures: Structure mapping dictionary.

        Returns:
            Tuple of (section_counts, section_colocalization).
        """
        # Load data
        with open(pred_file, "rb") as f:
            predictions = pickle.load(f)
        with open(anno_file, "rb") as f:
            annotation = pickle.load(f)

        # Initialize counts
        counts = {c: {} for c in range(len(predictions))}
        boxes = {c: [] for c in range(len(predictions))}

        # Process each channel
        for channel, detection in enumerate(predictions):
            counts[channel] = self._count_channel(detection, annotation, structures)
            boxes[channel] = detection.boxes

        # Compute colocalization
        colocalization = {}
        for c1, boxes1 in boxes.items():
            colocalization[c1] = {}
            for c2, boxes2 in boxes.items():
                colocalization[c1][c2] = compute_colocalization(boxes1, boxes2)

        return counts, colocalization
