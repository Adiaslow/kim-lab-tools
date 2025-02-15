# src/kim_lab_tools/application/use_cases/analyzers/roi_area_analyzer.py
"""
ROI Area Analyzer module for computing areas of regions of interest from pickle files.

This module provides functionality to analyze the area of ROIs across multiple brain segments
and regions, following the structure of the Allen Brain Atlas.
"""

# Standard Library Imports
import os
import sys
import platform
import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Third Party Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger: logging.Logger = logging.getLogger(__name__)


# Constants
BATCH_SIZE = 1
BUFFER_SIZE = 1024 * 1024
ROI_SIZE_THRESHOLD = 5000


def detect_gpu_backend() -> Tuple[Optional[str], Optional[Any]]:
    """
    Detect available GPU backend based on platform and hardware.

    Returns:
        Tuple of (backend_name, backend_module) or (None, None) if no GPU available.
    """
    system: str = platform.system()
    machine: str = platform.machine()

    # Try CuPy first (NVIDIA GPUs)
    try:
        import cupy as cp  # type: ignore

        return "cupy", cp
    except ImportError:
        pass

    # Try Metal Performance Shaders (Apple Silicon)
    if system == "Darwin" and machine == "arm64":
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps", torch
        except ImportError:
            pass

    return None, None


@dataclass
class ROIAreaResult:
    """Data class for storing ROI area analysis results.

    Attributes:
        segment_id: ID of the brain segment
        region_name: Name of the brain region
        area_pixels: Area of the ROI in pixels
    """

    segment_id: str
    region_name: str
    area_pixels: int


class ROIAreaAnalyzer:
    """Analyzer for computing areas of ROIs from pickle files.

    This class provides methods to:
    1. Process ROI data from pickle files
    2. Compute areas using different methods (fast, GPU, sparse) based on data size
    3. Generate summaries and statistics
    4. Track computation method usage

    Attributes:
        input_dir: Directory containing ROI pickle files
        max_workers: Maximum number of worker threads for parallel processing
        use_gpu: Whether to use GPU acceleration if available
        gpu_backend: GPU backend to use (cupy, mps, or None)
        gpu_module: GPU module to use (cupy or torch)
        method_counts: Dictionary tracking computation method usage
        _cache: Dictionary caching processed ROI data

    Methods:
        __init__: Initialize the ROI area analyzer
        analyze_directory: Analyze all ROI files in the input directory
        _compute_roi_area_fast: Compute area by counting dictionary keys (fastest method)
        _compute_roi_area_gpu: Compute area using available GPU acceleration
        _compute_roi_area_sparse: Compute area using memory-efficient sparse matrix representation
        _compute_roi_area: Compute area of ROI in pixels using the most appropriate method
        _parse_filename: Parse ROI filename into components
        _read_pickle_file: Read pickle file with optimized buffering
        _process_single_file: Process a single ROI file
        _process_batch: Process a batch of files in parallel, using GPU batch processing when possible
        _process_batch_gpu: Process large ROIs on GPU in a single batch
        _get_file_batches: Get batches of files to process
        analyze_directory: Analyze all ROI files in the input directory
        _optimize_dataframe: Optimize DataFrame memory usage
        get_summary_by_region: Get summary statistics by region
        get_summary_by_segment: Get summary statistics by segment
        clear_cache: Clear the cache of processed ROI data
        get_system_info: Get system information and GPU details
    """

    def __init__(
        self,
        input_dir: str,
        max_workers: Optional[int] = None,
        use_gpu: bool = False,
    ) -> None:
        """Initialize the ROI area analyzer.

        Args:
            input_dir: Directory containing ROI pickle files
            max_workers: Maximum number of worker threads for parallel processing
            use_gpu: Whether to use GPU acceleration if available
        """
        self.input_dir: Path = Path(input_dir)
        self.max_workers: int = (
            max_workers if max_workers is not None else os.cpu_count()  # type: ignore
        )
        self._cache: Dict[str, Any] = {}

        # GPU initialization
        self.use_gpu: bool = use_gpu
        if use_gpu:
            gpu_backend: Optional[str] = None
            gpu_module: Optional[Any] = None
            gpu_backend, gpu_module = detect_gpu_backend()
            self.gpu_backend: Optional[str] = gpu_backend
            self.gpu_module: Optional[Any] = gpu_module
            if self.gpu_backend:
                logger.info(f"Using GPU acceleration with {self.gpu_backend}")
                if self.gpu_backend == "cupy":
                    device: Any = self.gpu_module.cuda.runtime.getDeviceProperties(0)  # type: ignore
                    logger.info(f"GPU Device: {device['name'].decode()}")
                elif self.gpu_backend == "mps":
                    logger.info("GPU Device: Apple Silicon")
            else:
                logger.warning("No GPU backend available. Falling back to CPU.")
                self.use_gpu = False
        else:
            self.gpu_backend = None
            self.gpu_module = None
            logger.info("Using CPU computation (GPU acceleration disabled)")

        # Initialize method tracking
        self.method_counts: Dict[str, int] = {"fast": 0, "gpu": 0, "sparse": 0}

    def _compute_roi_area_fast(
        self,
        roi_data: Dict[str, Any],
    ) -> int:
        """Compute area by counting dictionary keys (fastest method).

        This method is most efficient for small ROIs or when intensity values
        don't need to be considered.

        Args:
            roi_data: Dictionary containing ROI data

        Returns:
            int: Area of the ROI in pixels
        """
        self.method_counts["fast"] += 1
        return len(roi_data["roi"])

    def _compute_roi_area_gpu(
        self,
        roi_data: Dict[str, Any],
        threshold: int = 0,
    ) -> int:
        """Compute area using available GPU acceleration.
        Optimized for batch processing and GPU utilization.

        Args:
            roi_data: Dictionary containing ROI data
            threshold: Threshold value for pixel intensity

        Returns:
            int: Area of the ROI in pixels
        """
        self.method_counts["gpu"] += 1
        values: np.ndarray = np.array(list(roi_data["roi"].values()), dtype=np.uint8)

        if not self.use_gpu:
            return int(np.count_nonzero(values > threshold))

        if self.gpu_backend == "cupy":
            # Process in larger chunks for better GPU utilization
            try:
                # Transfer to GPU
                logger.debug(f"Transferring {values.shape[0]} values to GPU")
                values_gpu: Any = self.gpu_module.asarray(values)  # type: ignore

                # Create mask on GPU
                mask_gpu: Any = values_gpu > threshold

                # Compute result on GPU
                area: int = int(self.gpu_module.sum(mask_gpu))  # type: ignore

                # Clean up GPU memory
                del values_gpu
                del mask_gpu
                self.gpu_module.get_default_memory_pool().free_all_blocks()  # type: ignore

                return area

            except Exception as e:
                logger.warning(f"GPU processing failed: {str(e)}. Falling back to CPU.")
                return int(np.count_nonzero(values > threshold))

        elif self.gpu_backend == "mps":
            try:
                # Convert to torch tensor and move to MPS
                logger.debug(f"Processing {values.shape[0]} values on MPS")
                values_gpu: Any = self.gpu_module.from_numpy(values).to("mps")  # type: ignore

                # Process on GPU
                mask_gpu: Any = values_gpu > threshold
                area: int = int(mask_gpu.sum().item())  # type: ignore

                # Clean up
                del values_gpu
                del mask_gpu
                self.gpu_module.mps.empty_cache()  # type: ignore

                return area

            except Exception as e:
                logger.warning(f"MPS processing failed: {str(e)}. Falling back to CPU.")
                return int(np.count_nonzero(values > threshold))

        return int(np.count_nonzero(values > threshold))

    def _compute_roi_area_sparse(self, roi_data: Dict[str, Any]) -> int:
        """Compute area using memory-efficient sparse matrix representation.

        This method is optimal for large ROIs when GPU is not available, as it
        minimizes memory usage through sparse matrix representation.

        Args:
            roi_data: Dictionary containing ROI data

        Returns:
            int: Area of the ROI in pixels
        """
        self.method_counts["sparse"] += 1
        coords: np.ndarray = np.array(list(roi_data["roi"].keys()))
        values: np.ndarray = np.array(list(roi_data["roi"].values()), dtype=np.uint8)

        y_max: int = int(coords.max(axis=0)[0]) + 1
        x_max: int = int(coords.max(axis=0)[1]) + 1
        sparse_mat: sparse.coo_matrix = sparse.coo_matrix(
            (values, (coords[:, 0], coords[:, 1])),
            shape=(y_max, x_max),
            dtype=np.uint8,
        )
        return int((sparse_mat.data > 0).sum())

    def _compute_roi_area(self, roi_data: Dict[str, Any]) -> int:
        """Compute area of ROI in pixels using the most appropriate method.

        The method selection is based on:
        1. ROI size (small ROIs use fast method)
        2. GPU availability (large ROIs use GPU if available)
        3. Memory efficiency (falls back to sparse method for large ROIs without GPU)

        Args:
            roi_data: Dictionary containing ROI data

        Returns:
            int: Area of the ROI in pixels
        """
        if not isinstance(roi_data, dict) or "roi" not in roi_data:
            raise ValueError("ROI data must be a dictionary containing 'roi' key")

        roi_coords: Dict[str, Any] = roi_data["roi"]
        if not isinstance(roi_coords, dict):
            raise ValueError("ROI coordinates must be a dictionary")

        roi_size: int = len(roi_coords)

        # For small ROIs, use fast method
        if roi_size < ROI_SIZE_THRESHOLD:
            logger.debug(f"Using fast method for ROI with {roi_size} pixels")
            return self._compute_roi_area_fast(roi_data)

        # For large ROIs, use GPU if available
        if self.use_gpu:
            logger.debug(f"Using GPU method for ROI with {roi_size} pixels")
            return self._compute_roi_area_gpu(roi_data)

        # Otherwise, use sparse matrix for memory efficiency
        logger.debug(f"Using sparse matrix method for ROI with {roi_size} pixels")
        return self._compute_roi_area_sparse(roi_data)

    @lru_cache(maxsize=1024)
    def _parse_filename(
        self,
        filename: str,
    ) -> Tuple[str, str, str]:
        """Parse ROI filename into components.

        Args:
            filename: Name of the ROI file

        Returns:
            Tuple of (animal_id, segment_id, region_name)
        """
        base: str = os.path.splitext(filename)[0]
        parts: List[str] = base.split("_")
        return parts[0], parts[1], "_".join(parts[2:])

    def _read_pickle_file(
        self,
        file_path: Path,
    ) -> Any:
        """Read pickle file with optimized buffering.

        Args:
            file_path: Path to the pickle file

        Returns:
            Dictionary containing ROI data
        """
        with open(file_path, "rb", buffering=BUFFER_SIZE) as f:
            return pickle.load(f)

    def _process_single_file(
        self,
        file: Path,
    ) -> Optional[Dict[str, Any]]:
        """Process a single ROI file.

        Args:
            file: Path to the ROI file

        Returns:
            Dictionary containing processed ROI data
        """
        try:
            # Use cached result if available
            cache_key = str(file)
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Parse filename first to avoid loading file if filename is invalid
            animal_id: str
            segment_id: str
            region_name: str
            animal_id, segment_id, region_name = self._parse_filename(file.name)

            roi_data: Dict[str, Any] = self._read_pickle_file(file)
            area: int = self._compute_roi_area(roi_data)

            result: Dict[str, Any] = {
                "animal_id": animal_id,
                "segment_id": segment_id,
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

    def _process_batch(
        self,
        files: List[Path],
    ) -> List[Dict[str, Any]]:
        """Process a batch of files in parallel, using GPU batch processing when possible.

        Args:
            files: List of paths to the ROI files

        Returns:
            List of dictionaries containing processed ROI data
        """
        try:
            # First, load all ROI data
            roi_data_list: List[Dict[str, Any]] = []
            file_info_list: List[Tuple[Path, str, str, str]] = []

            for file in files:
                try:
                    # Use cached result if available
                    cache_key = str(file)
                    if cache_key in self._cache:
                        return [self._cache[cache_key]]

                    # Parse filename
                    animal_id: str
                    segment_id: str
                    region_name: str
                    animal_id, segment_id, region_name = self._parse_filename(file.name)
                    roi_data: Dict[str, Any] = self._read_pickle_file(file)

                    # Store data and file info
                    roi_data_list.append(roi_data)
                    file_info_list.append((file, animal_id, segment_id, region_name))

                except Exception as e:
                    logger.error(f"Error loading {file.name}: {str(e)}")
                    continue

            if not roi_data_list:
                return []

            # Process ROIs in batch if they're large enough
            large_rois: List[Dict[str, Any]] = [
                roi for roi in roi_data_list if len(roi["roi"]) >= ROI_SIZE_THRESHOLD
            ]
            small_rois: List[Dict[str, Any]] = [
                roi for roi in roi_data_list if len(roi["roi"]) < ROI_SIZE_THRESHOLD
            ]

            results: List[Dict[str, Any]] = []

            # Process large ROIs in GPU batch if available
            if large_rois and self.use_gpu:
                logger.debug(f"Processing {len(large_rois)} large ROIs in GPU batch")
                areas: List[int] = self._process_batch_gpu(large_rois)

                # Match areas with file info
                for roi_data, area, (file, animal_id, segment_id, region_name) in zip(
                    large_rois, areas, file_info_list[: len(large_rois)]
                ):
                    result: Dict[str, Any] = {
                        "animal_id": animal_id,
                        "segment_id": segment_id,
                        "region_name": region_name,
                        "area_pixels": area,
                        "file_path": str(file),
                    }
                    self._cache[str(file)] = result
                    results.append(result)

            # Process remaining ROIs with regular methods
            remaining_files: List[Tuple[Path, str, str, str]] = file_info_list[
                len(large_rois) :
            ]
            remaining_rois: List[Dict[str, Any]] = small_rois

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_roi = {}
                for roi_data, (file, animal_id, segment_id, region_name) in zip(
                    remaining_rois, remaining_files
                ):
                    future = executor.submit(self._compute_roi_area, roi_data)
                    future_to_roi[future] = (file, animal_id, segment_id, region_name)

                for future in as_completed(future_to_roi):
                    try:
                        area: int = future.result()
                        file: Path
                        animal_id: str
                        segment_id: str
                        region_name: str
                        file, animal_id, segment_id, region_name = future_to_roi[future]
                        result: Dict[str, Any] = {
                            "animal_id": animal_id,
                            "segment_id": segment_id,
                            "region_name": region_name,
                            "area_pixels": area,
                            "file_path": str(file),
                        }
                        self._cache[str(file)] = result
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing ROI: {str(e)}")

            return results

        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            return []

    def _process_batch_gpu(
        self,
        roi_data_list: List[Dict[str, Any]],
    ) -> List[int] | None:
        """Process multiple ROIs on GPU in a single batch for better efficiency.

        Args:
            roi_data_list: List of dictionaries containing ROI data

        Returns:
            List of areas of the ROIs in pixels
        """
        try:
            # Extract all values and combine into a single array
            all_values: List[np.ndarray] = [
                np.array(list(data["roi"].values()), dtype=np.uint8)
                for data in roi_data_list
            ]
            total_values: int = sum(len(v) for v in all_values)
            logger.debug(
                f"Processing batch of {len(roi_data_list)} ROIs with {total_values} total values"
            )

            if self.gpu_backend == "mps":
                # Process batch on MPS
                batch_results: List[int] = []
                values_gpu = None

                for values in all_values:
                    if values_gpu is None:
                        values_gpu: Any = self.gpu_module.from_numpy(values).to("mps")  # type: ignore
                    else:
                        values_gpu: Any = self.gpu_module.cat(  # type: ignore
                            [values_gpu, self.gpu_module.from_numpy(values).to("mps")]  # type: ignore
                        )

                    mask_gpu = values_gpu > 0
                    area = int(mask_gpu.sum().item())
                    batch_results.append(area)

                # Clean up
                del values_gpu
                self.gpu_module.mps.empty_cache()  # type: ignore

                return batch_results

            elif self.gpu_backend == "cupy":
                # Process batch on CUDA
                batch_results: List[int] = []
                for values in all_values:
                    values_gpu: Any = self.gpu_module.asarray(values)  # type: ignore
                    area: int = int(self.gpu_module.count_nonzero(values_gpu))  # type: ignore
                    batch_results.append(area)
                    del values_gpu

                self.gpu_module.get_default_memory_pool().free_all_blocks()  # type: ignore
                return batch_results

        except Exception as e:
            logger.warning(
                f"Batch GPU processing failed: {str(e)}. Processing individually."
            )
            return [self._compute_roi_area_gpu(data) for data in roi_data_list]

    def _get_file_batches(self) -> Iterator[List[Path]]:
        """Get batches of files to process.

        Returns:
            Iterator of batches of file paths
        """
        pkl_files = list(self.input_dir.glob("*.pkl"))
        for i in range(0, len(pkl_files), BATCH_SIZE):
            yield pkl_files[i : i + BATCH_SIZE]

    def analyze_directory(self) -> pd.DataFrame:
        """
        Analyze all ROI files in the input directory.

        Returns:
            DataFrame containing analysis results with columns:
            - animal_id: ID of the animal
            - segment_id: ID of the brain segment
            - region_name: Name of the brain region
            - area_pixels: Area of the ROI in pixels
            - file_path: Path to the source file
        """
        # Reset method counts
        self.method_counts = {"fast": 0, "gpu": 0, "sparse": 0}

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
            self._optimize_dataframe(df)

        # Log computation method statistics
        logger.info("\nComputation method usage statistics:")
        logger.info(f"Fast method used: {self.method_counts['fast']} times")
        logger.info(f"GPU method used: {self.method_counts['gpu']} times")
        logger.info(f"Sparse method used: {self.method_counts['sparse']} times")

        return df

    @staticmethod
    def _optimize_dataframe(df: pd.DataFrame) -> None:
        """Optimize DataFrame memory usage."""
        # Convert string columns to categorical
        for col in ["animal_id", "segment_id", "region_name"]:
            df[col] = df[col].astype("category")

        # Convert numeric columns to appropriate types
        df["area_pixels"] = df["area_pixels"].astype(np.int32)

    def get_summary_by_region(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics grouped by region."""
        if df is None:
            df = self.analyze_directory()

        with tqdm(total=1, desc="Generating region summary") as pbar:
            summary: pd.DataFrame = (
                df.groupby("region_name", observed=True)
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
            # Name the columns
            summary.columns = ["count", "mean", "std", "min", "max", "q25", "q75"]
            # Reset index to make region_name the first column
            summary = summary.reset_index()
            pbar.update(1)
        return summary

    def get_summary_by_segment(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get summary statistics grouped by segment."""
        if df is None:
            df = self.analyze_directory()

        with tqdm(total=1, desc="Generating segment summary") as pbar:
            summary: pd.DataFrame = (
                df.groupby("segment_id", observed=True)
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
            # Name the columns
            summary.columns = ["count", "mean", "std", "min", "max", "q25", "q75"]
            # Reset index to make segment_id the first column
            summary = summary.reset_index()
            pbar.update(1)
        return summary

    def clear_cache(self) -> None:
        """Clear the internal cache of processed results."""
        self._cache.clear()
        self._parse_filename.cache_clear()

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the system and available GPU backends.

        Returns:
            Dictionary containing:
            - System information (OS, architecture)
            - GPU backend and device details
            - Computation method usage statistics
        """
        info: Dict[str, Any] = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "gpu_backend": self.gpu_backend,
            "using_gpu": self.use_gpu,
            "computation_methods": self.method_counts,
        }

        # Add GPU device info if available
        if self.use_gpu and self.gpu_backend == "cupy":
            device: Any = self.gpu_module.cuda.runtime.getDeviceProperties(0)  # type: ignore
            info["gpu_device"] = device["name"].decode()
        elif self.use_gpu and self.gpu_backend == "mps":
            info["gpu_device"] = "Apple Silicon"

        return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze ROI areas from pickle files")
    parser.add_argument("input_dir", help="Directory containing ROI pickle files")
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU acceleration if available"
    )
    parser.add_argument("--workers", type=int, help="Number of worker threads")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the logging level",
    )
    args: argparse.Namespace = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize analyzer
        analyzer = ROIAreaAnalyzer(
            input_dir=args.input_dir, max_workers=args.workers, use_gpu=args.use_gpu
        )

        # Print system info
        print("\nSystem Information:")
        print("-" * 20)
        sys_info: Dict[str, Any] = analyzer.get_system_info()
        for key, value in sys_info.items():
            if key != "computation_methods":
                print(f"{key}: {value}")

        # Run analysis
        print("\nAnalyzing ROIs...")
        df: pd.DataFrame = analyzer.analyze_directory()

        if df.empty:
            print("No ROIs found to analyze!")
            sys.exit(1)

        # Print computation statistics
        print("\nComputation Method Usage:")
        print("-" * 20)
        for method, count in analyzer.method_counts.items():
            print(f"{method.capitalize()} method: {count} ROIs")

        # Print basic statistics
        print("\nAnalysis Results:")
        print("-" * 20)
        print(f"Total ROIs analyzed: {len(df)}")
        print(f"Number of unique regions: {df['region_name'].nunique()}")
        print(f"Number of segments: {df['segment_id'].nunique()}")

        sys.exit(0)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
