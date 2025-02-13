"""Type definitions for axon segmentation."""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class SegmentationMetrics:
    """Metrics for segmentation evaluation."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float


@dataclass
class SegmentationBatch:
    """Batch data for segmentation."""

    images: Any
    masks: Any
    metadata: Dict[str, Any]
