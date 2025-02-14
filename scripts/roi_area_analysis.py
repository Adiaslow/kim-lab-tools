# examples/roi_area_analysis.py
"""
Example script demonstrating how to use the ROI Area Analyzer.

This script shows how to:
1. Load and analyze ROI data from pickle files
2. Generate different types of summaries
3. Save results to CSV files
4. Create basic visualizations

Example usage:
    python examples/roi_area_analysis.py /path/to/roi/directory
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add the src directory to Python path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.kim_lab_tools.application.use_cases.analyzers.roi_area_analyzer import (
    ROIAreaAnalyzer,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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
    return parser.parse_args()


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Create basic visualizations of the analysis results.

    Args:
        df: DataFrame containing analysis results
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn")

    with tqdm(total=2, desc="Creating visualizations") as pbar:
        # 1. Box plot of areas by region
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="region_name", y="area_pixels")
        plt.xticks(rotation=45, ha="right")
        plt.title("Distribution of ROI Areas by Region")
        plt.tight_layout()
        plt.savefig(output_dir / "area_by_region_boxplot.png", dpi=300)
        plt.close()
        pbar.update(1)

        # 2. Area trends across sections
        plt.figure(figsize=(12, 6))
        section_means = (
            df.groupby(["section_id", "region_name"])["area_pixels"].mean().unstack()
        )
        section_means.plot(marker="o")
        plt.title("ROI Area Trends Across Sections")
        plt.xlabel("Section ID")
        plt.ylabel("Mean Area (pixels)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_dir / "area_trends.png", dpi=300)
        plt.close()
        pbar.update(1)


def save_results(
    results_df: pd.DataFrame,
    region_summary: pd.DataFrame,
    section_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save analysis results to CSV files.

    Args:
        results_df: DataFrame containing detailed results
        region_summary: DataFrame containing summary by region
        section_summary: DataFrame containing summary by section
        output_dir: Directory to save results
    """
    with tqdm(total=3, desc="Saving results") as pbar:
        # Save detailed results
        results_df.to_csv(output_dir / "detailed_results.csv", index=False)
        pbar.update(1)

        # Save region summary
        region_summary.to_csv(output_dir / "region_summary.csv")
        pbar.update(1)

        # Save section summary
        section_summary.to_csv(output_dir / "section_summary.csv")
        pbar.update(1)


def print_statistics(results_df: pd.DataFrame) -> None:
    """Print basic statistics about the analysis results.

    Args:
        results_df: DataFrame containing analysis results
    """
    print("\nQuick Statistics:")
    print("-----------------")
    print(f"Total ROIs analyzed: {len(results_df)}")
    print(f"Number of unique regions: {results_df['region_name'].nunique()}")
    print(f"Number of sections: {results_df['section_id'].nunique()}")
    print("\nMean area by region:")
    print(results_df.groupby("region_name")["area_pixels"].mean().round(2).to_string())


def main() -> int:
    """Main function demonstrating ROI area analysis."""
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    try:
        # Initialize analyzer with the parent directory if input is a file
        analyzer = ROIAreaAnalyzer(
            str(input_path.parent if input_path.is_file() else input_path),
            max_workers=args.max_workers,
        )

        # Get detailed results
        print("\nAnalyzing ROIs...")
        results_df = analyzer.analyze_directory()

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
        region_summary = analyzer.get_summary_by_region(results_df)
        section_summary = analyzer.get_summary_by_section(results_df)

        # Save results
        save_results(results_df, region_summary, section_summary, output_dir)
        print(f"\nResults saved to: {output_dir}/")

        # Create visualizations if we have multiple ROIs
        if len(results_df) > 1:
            create_visualizations(results_df, output_dir)

        # Print statistics
        print_statistics(results_df)

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
