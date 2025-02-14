"""CLI for training data preparation."""

import argparse
from pathlib import Path

from ..processors.data_processor import DataProcessor


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Create training data")
    parser.add_argument(
        "--images",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Directory containing label images",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tiles",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Size of output tiles",
    )
    args = parser.parse_args()

    # Initialize processor
    processor = DataProcessor(tile_size=args.tile_size)

    # Process directories
    processor.process_directory(
        Path(args.images),
        Path(args.labels),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
