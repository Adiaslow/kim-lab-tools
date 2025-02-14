# src/application/use_cases/analyzers/roi_area_analyzer.py
"""
ROI Area Analyzer module for computing areas of regions of interest from pickle files.

This module provides functionality to analyze the area of ROIs across multiple brain sections
and regions, following the structure of the Allen Brain Atlas.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


@dataclass
class ROIAreaResult:
    """Data class for storing ROI area analysis results."""

    section_id: str
    region_name: str
    area_pixels: int


class ROIAreaAnalyzer:
    """Analyzer for computing areas of ROIs from pickle files.

    This analyzer processes ROI data saved as pickle files, where each file contains:
    - A dictionary with 'roi' and 'name' keys
    - The 'roi' value is a dictionary where:
        - Keys are (y, x) coordinate tuples as np.int64
        - Values are uint8 intensity values
    """

    def __init__(self, input_dir: str, max_workers: int = None):
        """Initialize the ROI area analyzer.

        Args:
            input_dir: Directory containing ROI pickle files
            max_workers: Maximum number of worker threads for parallel processing.
                       If None, uses the default from ThreadPoolExecutor.
        """
        self.input_dir = Path(input_dir)
        self.max_workers = max_workers

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

    def _compute_roi_area(self, roi_data: Dict[str, Any]) -> int:
        """Compute area of ROI in pixels.

        Args:
            roi_data: Dictionary containing ROI data with keys:
                - 'roi': Dictionary of (y,x) coordinate tuples to intensity values
                - 'name': Name of the ROI

        Returns:
            Area in pixels (number of coordinates in the ROI)

        Raises:
            ValueError: If ROI data format is invalid
        """
        if not isinstance(roi_data, dict) or "roi" not in roi_data:
            raise ValueError("ROI data must be a dictionary containing 'roi' key")

        roi_coords = roi_data["roi"]
        if not isinstance(roi_coords, dict):
            raise ValueError("ROI coordinates must be a dictionary")

        return len(roi_coords)

    def _process_single_file(self, file: Path) -> Optional[Dict[str, Any]]:
        """Process a single ROI file.

        Args:
            file: Path to the ROI file

        Returns:
            Dictionary containing analysis results or None if processing failed
        """
        try:
            animal_id, section_id, region_name = self._parse_filename(file.name)

            with open(file, "rb") as f:
                roi_data = pickle.load(f)

            area = self._compute_roi_area(roi_data)

            return {
                "animal_id": animal_id,
                "section_id": section_id,
                "region_name": region_name,
                "area_pixels": area,
                "file_path": str(file),
            }
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
            return None

    def analyze_directory(self) -> pd.DataFrame:
        """Analyze all ROI files in the input directory.

        Returns:
            DataFrame containing analysis results with columns:
            - animal_id: ID of the animal
            - section_id: ID of the brain section
            - region_name: Name of the brain region
            - area_pixels: Area of the ROI in pixels
            - file_path: Path to the source file

        Note:
            Area is computed as the number of coordinates in the ROI dictionary,
            which represents the number of pixels in the ROI mask.
        """
        pkl_files = list(self.input_dir.glob("*.pkl"))
        if not pkl_files:
            print(f"No .pkl files found in {self.input_dir}")
            return pd.DataFrame()

        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files for processing
            future_to_file = {
                executor.submit(self._process_single_file, file): file
                for file in pkl_files
            }

            # Process results as they complete with progress bar
            with tqdm(total=len(pkl_files), desc="Processing ROI files") as pbar:
                for future in as_completed(future_to_file):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    pbar.update(1)

        # Create DataFrame and optimize types
        df = pd.DataFrame(results)
        if not df.empty:
            # Optimize memory usage by converting to categorical where appropriate
            for col in ["animal_id", "section_id", "region_name"]:
                df[col] = df[col].astype("category")

            # Ensure area_pixels is integer type
            df["area_pixels"] = df["area_pixels"].astype(np.int32)

        return df

    def get_summary_by_region(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics grouped by region.

        Args:
            df: Optional DataFrame from analyze_directory(). If None, will run analysis.

        Returns:
            DataFrame with summary statistics by region
        """
        if df is None:
            df = self.analyze_directory()

        with tqdm(total=1, desc="Generating region summary") as pbar:
            summary = (
                df.groupby("region_name")
                .agg(
                    {
                        "area_pixels": [
                            "count",
                            "mean",
                            "std",
                            "min",
                            "max",
                            lambda x: x.quantile(0.25),
                            lambda x: x.quantile(0.75),
                        ]
                    }
                )
                .round(2)
            )
            summary.columns = ["count", "mean", "std", "min", "max", "q25", "q75"]
            pbar.update(1)

        return summary

    def get_summary_by_section(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics grouped by section.

        Args:
            df: Optional DataFrame from analyze_directory(). If None, will run analysis.

        Returns:
            DataFrame with summary statistics by section
        """
        if df is None:
            df = self.analyze_directory()

        with tqdm(total=1, desc="Generating section summary") as pbar:
            summary = (
                df.groupby("section_id")
                .agg(
                    {
                        "area_pixels": [
                            "count",
                            "mean",
                            "std",
                            "min",
                            "max",
                            lambda x: x.quantile(0.25),
                            lambda x: x.quantile(0.75),
                        ]
                    }
                )
                .round(2)
            )
            summary.columns = ["count", "mean", "std", "min", "max", "q25", "q75"]
            pbar.update(1)

        return summary
