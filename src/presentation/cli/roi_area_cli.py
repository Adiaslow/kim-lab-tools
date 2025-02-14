# src/presentation/cli/roi_area_cli.py

"""
Command-line interface for ROI area analysis.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ...application.use_cases.analyzers.roi_area_analyzer import ROIAreaAnalyzer


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Analyze ROI areas from pickle files")
    parser.add_argument(
        "input_dir", type=str, help="Directory containing ROI pickle files"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output CSV file path (optional)"
    )
    parser.add_argument(
        "--summary-by",
        choices=["region", "section"],
        help="Generate summary statistics grouped by region or section",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point for ROI area analysis CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()

    try:
        # Initialize analyzer
        analyzer = ROIAreaAnalyzer(args.input_dir)

        # Run analysis
        results = analyzer.analyze_directory()

        # Generate summary if requested
        if args.summary_by == "region":
            summary = analyzer.get_summary_by_region(results)
            results = summary
        elif args.summary_by == "section":
            summary = analyzer.get_summary_by_section(results)
            results = summary

        # Save or display results
        if args.output:
            results.to_csv(args.output)
            print(f"Results saved to: {args.output}")
        else:
            print("\nAnalysis Results:")
            print("----------------")
            print(results.to_string())

        return 0

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
