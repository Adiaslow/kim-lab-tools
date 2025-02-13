"""Processor for axon segmentation inference."""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from ..core.base import BaseProcessor
from ..models.segformer_model import AxonSegmentationModel


class AxonSegmentationProcessor(BaseProcessor):
    """Processor for running axon segmentation inference."""

    # ... inference processor implementation ...
