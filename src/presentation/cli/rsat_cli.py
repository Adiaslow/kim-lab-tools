"""CLI for RSAT segmentation."""

import argparse
from pathlib import Path

from ..processors.rsat_processor import RSATProcessor


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RSAT image segmentation")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output image path",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        required=True,
        help="Tile size in pixels",
    )
    args = parser.parse_args()

    # Initialize processor
    processor = RSATProcessor(tile_size=args.size)

    # Process image
    processor.process_file(
        Path(args.input),
        Path(args.output),
    )


if __name__ == "__main__":
    main()
