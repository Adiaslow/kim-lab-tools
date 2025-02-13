"""CLI for axon segmentation tasks."""

import argparse
from pathlib import Path

from ..processors.inference_processor import AxonSegmentationProcessor
from ..trainers.axon_segmentation_trainer import AxonSegmentationTrainer


def main() -> None:
    """Main entry point."""
    # ... CLI implementation ...
