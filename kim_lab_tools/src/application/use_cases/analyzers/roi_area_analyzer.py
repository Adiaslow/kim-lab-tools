# src/application/use_cases/analyzers/roi_area_analyzer.py
"""
ROI Area Analyzer module for computing areas of regions of interest from pickle files.

This module provides functionality to analyze the area of ROIs across multiple brain sections
and regions, following the structure of the Allen Brain Atlas.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class ROIAreaResult:
    """Data class for storing ROI area analysis results."""

    section_id: str
    region_name: str
    area_pixels: int


class ROIAreaAnalyzer:
    """Analyzer for computing areas of ROIs from pickle files."""

    def __init__(self, input_dir: str):
        """Initialize the ROI area analyzer.

        Args:
            input_dir: Directory containing ROI pickle files
        """
        self.input_dir = Path(input_dir)

    def _parse_filename(self, filename: str) -> Tuple[str, str, str]:
        """Parse ROI filename into components.

        Args:
            filename: Name of the ROI file (e.g., 'M762_s038_RSPagl.pkl')

        Returns:
            Tuple containing (animal_id, section_id, region_name)
        """
        base = os.path.splitext(filename)[0]
        parts = base.split("_")
        return parts[0], parts[1], "_".join(parts[2:])

    def _compute_roi_area(self, roi_data) -> int:
        """Compute area of ROI in pixels.

        Args:
            roi_data: ROI data loaded from pickle file

        Returns:
            Area in pixels
        """
        # Assuming ROI data is a binary mask or contains a mask
        # Modify this based on actual pickle file structure
        if isinstance(roi_data, np.ndarray):
            return int(np.sum(roi_data > 0))
        elif hasattr(roi_data, "mask"):
            return int(np.sum(roi_data.mask > 0))
        else:
            raise ValueError(f"Unsupported ROI data format: {type(roi_data)}")

    def analyze_directory(self) -> pd.DataFrame:
        """Analyze all ROI files in the input directory.

        Returns:
            DataFrame containing analysis results with columns:
            - animal_id: ID of the animal
            - section_id: ID of the brain section
            - region_name: Name of the brain region
            - area_pixels: Area of the ROI in pixels
        """
        results = []

        for file in sorted(self.input_dir.glob("*.pkl")):
            try:
                # Parse filename
                animal_id, section_id, region_name = self._parse_filename(file.name)

                # Load ROI data
                with open(file, "rb") as f:
                    roi_data = pickle.load(f)

                # Compute area
                area = self._compute_roi_area(roi_data)

                # Store result
                results.append(
                    {
                        "animal_id": animal_id,
                        "section_id": section_id,
                        "region_name": region_name,
                        "area_pixels": area,
                    }
                )

            except Exception as e:
                print(f"Error processing {file.name}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def get_summary_by_region(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics grouped by region.

        Args:
            df: Optional DataFrame from analyze_directory(). If None, will run analysis.

        Returns:
            DataFrame with summary statistics by region
        """
        if df is None:
            df = self.analyze_directory()

        return (
            df.groupby("region_name")
            .agg({"area_pixels": ["count", "mean", "std", "min", "max"]})
            .round(2)
        )

    def get_summary_by_section(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics grouped by section.

        Args:
            df: Optional DataFrame from analyze_directory(). If None, will run analysis.

        Returns:
            DataFrame with summary statistics by section
        """
        if df is None:
            df = self.analyze_directory()

        return (
            df.groupby("section_id")
            .agg({"area_pixels": ["count", "mean", "std", "min", "max"]})
            .round(2)
        )
