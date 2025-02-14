# src/application/use_cases/analyzers/roi_area_analyzer.py
"""
ROI Area Analyzer module for computing areas of regions of interest from pickle files.

This module provides functionality to analyze the area of ROIs across multiple brain sections
and regions, following the structure of the Allen Brain Atlas.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import lru_cache
import logging
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
BATCH_SIZE = 10
BUFFER_SIZE = 1024 * 1024  # 1MB buffer for file reading


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

    def __init__(self, input_dir: str, max_workers: Optional[int] = None):
        """Initialize the ROI area analyzer.

        Args:
            input_dir: Directory containing ROI pickle files
            max_workers: Maximum number of worker threads for parallel processing.
                       If None, uses the default from ThreadPoolExecutor.
        """
        self.input_dir = Path(input_dir)
        self.max_workers = max_workers if max_workers is not None else os.cpu_count()
        self._cache = {}

    @lru_cache(maxsize=1024)
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

    def _read_pickle_file(self, file_path: Path) -> Any:
        """Read pickle file with optimized buffering.

        Args:
            file_path: Path to the pickle file

        Returns:
            Unpickled data object
        """
        with open(file_path, "rb", buffering=BUFFER_SIZE) as f:
            return pickle.load(f)

    def _process_single_file(self, file: Path) -> Optional[Dict[str, Any]]:
        """Process a single ROI file.

        Args:
            file: Path to the ROI file

        Returns:
            Dictionary containing analysis results or None if processing failed
        """
        try:
            animal_id, section_id, region_name = self._parse_filename(file.name)

            # Use cached result if available
            cache_key = str(file)
            if cache_key in self._cache:
                return self._cache[cache_key]

            roi_data = self._read_pickle_file(file)
            area = self._compute_roi_area(roi_data)

            result = {
                "animal_id": animal_id,
                "section_id": section_id,
                "region_name": region_name,
                "area_pixels": area,
                "file_path": str(file),
            }

            # Cache the result
            self._cache[cache_key] = result
            return result

        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            return None

    def _process_batch(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Process a batch of files in parallel.

        Args:
            files: List of files to process

        Returns:
            List of processed results
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, file): file for file in files
            }
            for future in as_completed(future_to_file):
                result = future.result()
                if result is not None:
                    results.append(result)
        return results

    def _get_file_batches(self) -> Iterator[List[Path]]:
        """Get batches of files to process.

        Yields:
            Lists of file paths, with each list containing at most BATCH_SIZE files
        """
        pkl_files = list(self.input_dir.glob("*.pkl"))
        for i in range(0, len(pkl_files), BATCH_SIZE):
            yield pkl_files[i : i + BATCH_SIZE]

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
        if not self.input_dir.exists():
            logger.error(f"Directory not found: {self.input_dir}")
            return pd.DataFrame()

        pkl_files = list(self.input_dir.glob("*.pkl"))
        if not pkl_files:
            logger.warning(f"No .pkl files found in {self.input_dir}")
            return pd.DataFrame()

        all_results = []
        total_batches = (len(pkl_files) + BATCH_SIZE - 1) // BATCH_SIZE

        with tqdm(total=len(pkl_files), desc="Processing ROI files") as pbar:
            for batch in self._get_file_batches():
                batch_results = self._process_batch(batch)
                all_results.extend(batch_results)
                pbar.update(len(batch))

        # Create DataFrame and optimize types
        df = pd.DataFrame(all_results)
        if not df.empty:
            # Optimize memory usage
            self._optimize_dataframe(df)

        return df

    @staticmethod
    def _optimize_dataframe(df: pd.DataFrame) -> None:
        """Optimize DataFrame memory usage.

        Args:
            df: DataFrame to optimize
        """
        # Convert string columns to categorical
        for col in ["animal_id", "section_id", "region_name"]:
            df[col] = df[col].astype("category")

        # Convert numeric columns to appropriate types
        df["area_pixels"] = df["area_pixels"].astype(np.int32)

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
                df.groupby(
                    "region_name", observed=True
                )  # Add observed=True for categorical
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
                df.groupby(
                    "section_id", observed=True
                )  # Add observed=True for categorical
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

    def clear_cache(self) -> None:
        """Clear the internal cache of processed results."""
        self._cache.clear()
        self._parse_filename.cache_clear()
