"""CLI for generating synthetic experiments."""

import argparse
from pathlib import Path
import nrrd

from ..generators.experiment_generator import ExperimentGenerator


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic experiments from 3D volumes"
    )
    parser.add_argument(
        "-v", "--volume", help="Path to input volume file", required=True
    )
    parser.add_argument(
        "-o", "--output", help="Output directory for experiment", required=True
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        help="Number of samples to generate",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--size",
        help="Output image size (default: 224x224)",
        default="224x224",
    )
    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Parse size argument
    width, height = map(int, args.size.split("x"))

    # Load volume
    volume, _ = nrrd.read(str(Path(args.volume).expanduser()))

    # Generate experiment
    generator = ExperimentGenerator(
        output_path=Path(args.output),
        num_samples=args.num_samples,
        output_size=(width, height),
    )
    generator.process(volume)
    print("\nDone!")


if __name__ == "__main__":
    main()
