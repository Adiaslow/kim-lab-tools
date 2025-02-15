# tests/test_roi_area_analyzer.py
"""Test suite for the ROI Area Analyzer module."""

import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from kim_lab_tools.application.use_cases.analyzers.roi_area_analyzer import (
    ROIAreaAnalyzer,
    ROIAreaResult,
)


@pytest.fixture
def temp_roi_dir():
    """Create a temporary directory with sample ROI files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_roi_data():
    """Create sample ROI data with known properties."""
    # Create a simple 5x5 ROI with some zero and non-zero values
    coords = [
        (0, 0),
        (0, 1),
        (0, 2),  # First row
        (1, 1),  # Middle point
        (2, 0),
        (2, 1),
        (2, 2),  # Last row
    ]
    values = [1, 2, 1, 3, 1, 2, 1]  # All non-zero values

    roi_dict = {(y, x): val for (y, x), val in zip(coords, values)}
    return {"roi": roi_dict, "name": "test_roi"}


@pytest.fixture
def sample_roi_with_zeros():
    """Create sample ROI data with zero values to test filtering."""
    coords = [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    ]
    values = [1, 0, 1, 3, 0, 2, 1]  # Contains zero values

    roi_dict = {(y, x): val for (y, x), val in zip(coords, values)}
    return {"roi": roi_dict, "name": "test_roi_zeros"}


@pytest.fixture
def create_test_files(temp_roi_dir, sample_roi_data, sample_roi_with_zeros):
    """Create test ROI files in the temporary directory."""
    # Create multiple ROI files with different properties
    files_data = [
        ("M123_s001_Region1.pkl", sample_roi_data),
        ("M123_s002_Region1.pkl", sample_roi_data),
        ("M123_s001_Region2.pkl", sample_roi_with_zeros),
        ("M124_s001_Region1.pkl", sample_roi_with_zeros),
    ]

    created_files = []
    for filename, data in files_data:
        file_path = temp_roi_dir / filename
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        created_files.append(file_path)

    return created_files


def test_roi_area_result():
    """Test ROIAreaResult dataclass."""
    result = ROIAreaResult(section_id="s001", region_name="Region1", area_pixels=100)
    assert result.section_id == "s001"
    assert result.region_name == "Region1"
    assert result.area_pixels == 100


def test_analyzer_initialization():
    """Test ROIAreaAnalyzer initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        analyzer = ROIAreaAnalyzer(temp_dir)
        assert analyzer.input_dir == Path(temp_dir)
        assert analyzer.max_workers is not None
        assert analyzer._cache == {}


def test_filename_parsing():
    """Test ROI filename parsing."""
    analyzer = ROIAreaAnalyzer("dummy_path")
    animal_id, section_id, region_name = analyzer._parse_filename(
        "M123_s001_Region1.pkl"
    )

    assert animal_id == "M123"
    assert section_id == "s001"
    assert region_name == "Region1"


def test_compute_roi_area(sample_roi_data):
    """Test ROI area computation with non-zero values."""
    analyzer = ROIAreaAnalyzer("dummy_path")
    area = analyzer._compute_roi_area(sample_roi_data)

    # All values in sample_roi_data are non-zero, so area should be 7
    assert area == 7


def test_compute_roi_area_with_zeros(sample_roi_with_zeros):
    """Test ROI area computation with zero values."""
    analyzer = ROIAreaAnalyzer("dummy_path")
    area = analyzer._compute_roi_area(sample_roi_with_zeros)

    # Only count non-zero values (5 non-zero values in sample_roi_with_zeros)
    assert area == 5


def test_analyze_directory(create_test_files, temp_roi_dir):
    """Test full directory analysis."""
    analyzer = ROIAreaAnalyzer(str(temp_roi_dir))
    results_df = analyzer.analyze_directory()

    assert not results_df.empty
    assert len(results_df) == 4  # We created 4 test files
    assert set(results_df.columns) == {
        "animal_id",
        "section_id",
        "region_name",
        "area_pixels",
        "file_path",
    }

    # Check that animal IDs are correct
    assert set(results_df["animal_id"]) == {"M123", "M124"}

    # Check that section IDs are correct
    assert set(results_df["section_id"]) == {"s001", "s002"}

    # Check that region names are correct
    assert set(results_df["region_name"]) == {"Region1", "Region2"}


def test_get_summary_by_region(create_test_files, temp_roi_dir):
    """Test region summary generation."""
    analyzer = ROIAreaAnalyzer(str(temp_roi_dir))
    df = analyzer.analyze_directory()
    summary = analyzer.get_summary_by_region(df)

    assert not summary.empty
    assert "region_name" in summary.columns
    assert all(
        col in summary.columns
        for col in ["count", "mean", "std", "min", "max", "q25", "q75"]
    )


def test_get_summary_by_section(create_test_files, temp_roi_dir):
    """Test section summary generation."""
    analyzer = ROIAreaAnalyzer(str(temp_roi_dir))
    df = analyzer.analyze_directory()
    summary = analyzer.get_summary_by_section(df)

    assert not summary.empty
    assert "section_id" in summary.columns
    assert all(
        col in summary.columns
        for col in ["count", "mean", "std", "min", "max", "q25", "q75"]
    )


def test_cache_functionality(create_test_files, temp_roi_dir):
    """Test caching functionality."""
    analyzer = ROIAreaAnalyzer(str(temp_roi_dir))

    # First run should populate cache
    results1 = analyzer.analyze_directory()
    cache_size1 = len(analyzer._cache)

    # Second run should use cache
    results2 = analyzer.analyze_directory()
    cache_size2 = len(analyzer._cache)

    assert cache_size1 == cache_size2
    assert_array_equal(results1.values, results2.values)

    # Clear cache and verify
    analyzer.clear_cache()
    assert len(analyzer._cache) == 0


def test_error_handling():
    """Test error handling for invalid inputs."""
    analyzer = ROIAreaAnalyzer("nonexistent_directory")

    # Test analyzing non-existent directory
    results = analyzer.analyze_directory()
    assert results.empty

    # Test invalid ROI data
    with pytest.raises(ValueError):
        analyzer._compute_roi_area({"invalid": "data"})

    with pytest.raises(ValueError):
        analyzer._compute_roi_area({"roi": "invalid"})


def test_gpu_detection():
    """Test GPU detection and fallback."""
    analyzer = ROIAreaAnalyzer("dummy_path", use_gpu=True)

    # Check if GPU detection worked and fallback if not available
    assert hasattr(analyzer, "use_gpu")
    assert hasattr(analyzer, "gpu_backend")
    assert hasattr(analyzer, "gpu_module")

    # If no GPU is available, these should be None/False
    if not analyzer.gpu_backend:
        assert not analyzer.use_gpu
        assert analyzer.gpu_module is None


def test_method_counting(create_test_files, temp_roi_dir):
    """Test that method usage counting works correctly."""
    analyzer = ROIAreaAnalyzer(str(temp_roi_dir))

    # Reset method counts
    analyzer.method_counts = {"fast": 0, "gpu": 0, "sparse": 0}

    # Run analysis
    analyzer.analyze_directory()

    # Verify that methods were counted
    assert sum(analyzer.method_counts.values()) == 4  # We created 4 test files
    assert all(count >= 0 for count in analyzer.method_counts.values())
