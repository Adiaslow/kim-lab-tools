"""CLI for processing image stacks."""

import argparse
from pathlib import Path
import tifffile as tiff
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from ..processors.image_processor import ImageProcessor
from ..enhancers.contrast_enhancer import ContrastEnhancer


def process_file(
    file: Path,
    output_path: Path,
    equalize: bool = False,
    radius: float = 3.0,
    amount: float = 2.0,
) -> None:
    """Process a single image file.

    Args:
        file: Input file path.
        output_path: Output directory path.
        equalize: Whether to perform contrast equalization.
        radius: Radius for unsharp mask.
        amount: Amount for unsharp mask.
    """
    try:
        print(f"Processing {file}", flush=True)
        img = tiff.imread(file)

        if equalize:
            enhancer = ContrastEnhancer()
            img = enhancer.process(img)

        processor = ImageProcessor(radius=radius, amount=amount)
        img = processor.process(img)

        # Save processed image
        output_file = output_path / file.name
        tiff.imwrite(str(output_file), img)

    except Exception as e:
        print(f"Failed to process {file}. Error: {e}", flush=True)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Process z-stack images")
    parser.add_argument("-o", "--output", help="output directory", default="")
    parser.add_argument("-i", "--input", help="input directory", default="")
    parser.add_argument("-r", "--radius", help="radius for unsharp mask", default=3)
    parser.add_argument("-a", "--amount", help="amount for unsharp mask", default=2)
    parser.add_argument(
        "-e", "--equalize", help="equalize histogram", action="store_true"
    )
    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    input_path = Path(args.input.strip())
    output_path = Path(args.output.strip())
    output_path.mkdir(parents=True, exist_ok=True)

    amount = float(args.amount.strip())
    radius = float(args.radius.strip())

    valid_extensions = [".tif", ".tiff"]
    input_files = sorted(
        [file for file in input_path.iterdir() if file.suffix in valid_extensions]
    )
    print(f"Found {len(input_files)} files to process", flush=True)

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        fn = partial(
            process_file,
            output_path=output_path,
            equalize=args.equalize,
            radius=radius,
            amount=amount,
        )
        futures = [executor.submit(fn, file) for file in input_files]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
