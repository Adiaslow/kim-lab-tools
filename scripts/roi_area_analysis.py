# scripts/roi_area_analysis.py
"""
This script is used to analyze the area of ROIs in a directory of pickle files.

Usage:
    python scripts/roi_area_analysis.py <path_to_roi_pickle_file_or_directory>

Examples:
    ```bash
    python scripts/roi_area_analysis.py /path/to/roi/directory # using default settings
    python scripts/roi_area_analysis.py /path/to/roi/directory --max-workers 4 # using 4 worker threads
    python scripts/roi_area_analysis.py /path/to/roi/directory --use-gpu # using GPU if available
    python scripts/roi_area_analysis.py /path/to/roi/directory --output-dir /path/to/output/directory # save output to a directory
    ```
"""

# Standard Library Imports
import argparse
import sys
from pathlib import Path

# Third Party Imports
import pandas as pd
from typing import Optional

# Local Imports
from src.kim_lab_tools.application.use_cases.analyzers.roi_area_analyzer import (
    ROIAreaAnalyzer,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Example script for ROI area analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to ROI pickle file or directory containing ROI pickle files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save results (default: analysis_results)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker threads for parallel processing",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration if available",
    )
    return parser.parse_args()


def print_statistics(
    results_df: pd.DataFrame,
    analyzer: ROIAreaAnalyzer,
) -> None:
    """Print basic statistics about the analysis results.

    Args:
        results_df: DataFrame containing analysis results
        analyzer: ROIAreaAnalyzer instance with method statistics
    """
    # System information
    sys_info: dict = analyzer.get_system_info()
    print("\nSystem Information:")
    print("-----------------")
    print(f"System: {sys_info['system']} {sys_info['machine']}")
    print(f"GPU Backend: {sys_info['gpu_backend'] or 'None'}")
    if sys_info["using_gpu"]:
        print(f"GPU Device: {sys_info.get('gpu_device', 'Unknown')}")

    # Computation method statistics
    print("\nComputation Methods Used:")
    print("-----------------------")
    print(f"Fast method: {analyzer.method_counts['fast']} ROIs")
    print(f"GPU method: {analyzer.method_counts['gpu']} ROIs")
    print(f"Sparse method: {analyzer.method_counts['sparse']} ROIs")

    # General statistics
    print("\nROI Statistics:")
    print("--------------")
    print(f"Total ROIs analyzed: {len(results_df)}")
    print(f"Number of unique regions: {results_df['region_name'].nunique()}")
    print(f"Number of sections: {results_df['section_id'].nunique()}")
    print("\nMean area by region:")
    print(results_df.groupby("region_name")["area_pixels"].mean().round(2).to_string())


def save_results(
    results_df: pd.DataFrame,
    region_summary: pd.DataFrame,
    section_summary: pd.DataFrame,
    input_path: Path,
    output_dir: Optional[Path] = None,
) -> None:
    """Save the analysis results and summaries to CSV files.

    Args:
        results_df: DataFrame containing analysis results
        region_summary: DataFrame containing summary by region
        section_summary: DataFrame containing summary by section
        input_path: Path to the input file or directory
        output_dir: Optional path to the output directory. If not provided, results are saved in the input directory.
    """
    if not results_df.empty:
        # Use the output directory if provided, otherwise use the input directory
        save_dir = output_dir if output_dir else input_path
        results_path = save_dir / "roi_area_analysis.csv"
        region_summary_path = save_dir / "region_summary.csv"
        section_summary_path = save_dir / "section_summary.csv"

        results_df.to_csv(results_path, index=False)
        region_summary.to_csv(region_summary_path, index=False)
        section_summary.to_csv(section_summary_path, index=False)

        print(f"Results saved to: {results_path}")
        print(f"Region summary saved to: {region_summary_path}")
        print(f"Section summary saved to: {section_summary_path}")
    else:
        print("No results to save.")


def main() -> int:
    """Main function demonstrating ROI area analysis."""
    args: argparse.Namespace = parse_args()
    input_path: Path = Path(args.input_path)

    # Get base directory (directory containing pkl files)
    base_dir: Path = input_path.parent if input_path.is_file() else input_path
    # Set output directory using the base directory name
    output_dir: Path = base_dir.parent / f"{base_dir.name}_area_analysis"

    try:
        # Initialize analyzer with the same base directory logic
        analyzer = ROIAreaAnalyzer(
            str(input_path.parent if input_path.is_file() else input_path),
            max_workers=args.max_workers,
            use_gpu=args.use_gpu,
        )

        # Get detailed results
        print("\nAnalyzing ROIs...")
        results_df: pd.DataFrame = analyzer.analyze_directory()

        if results_df.empty:
            print("No ROIs found to analyze!")
            return 1

        # If input is a single file, filter results for that file
        if input_path.is_file():
            results_df = results_df[
                results_df.apply(
                    lambda x: f"{x['animal_id']}_s{x['section_id']}_{x['region_name']}.pkl"
                    == input_path.name,
                    axis=1,
                )
            ]

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get summaries
        print("\nGenerating summaries...")
        region_summary: pd.DataFrame = analyzer.get_summary_by_region(results_df)
        section_summary: pd.DataFrame = analyzer.get_summary_by_section(results_df)

        # Save results
        save_results(
            results_df, region_summary, section_summary, input_path, output_dir
        )
        print(f"\nResults saved to: {output_dir}/")

        # Print statistics with method tracking
        print_statistics(results_df, analyzer)

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
