"""CLI for axon detection evaluation."""

import argparse
from pathlib import Path

from ..processors.evaluation_processor import EvaluationProcessor


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate axon detection")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Label image path",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=5,
        help="Margin of error in pixels",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=50,
        help="Minimum object size",
    )
    args = parser.parse_args()

    # Initialize processor
    processor = EvaluationProcessor(
        margin=args.margin,
        min_size=args.min_size,
    )

    # Run evaluation
    processor.evaluate_file(
        Path(args.input),
        Path(args.label),
    )


if __name__ == "__main__":
    main()
