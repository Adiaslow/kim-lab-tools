"""CLI for z-stack image processing."""

import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import cv2

from ..processors.stack_processor import StackProcessor


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Process z-stack images")
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory",
        default="",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input directory",
        default="",
    )
    parser.add_argument(
        "-g",
        "--graphical",
        help="Use graphical interface",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-d",
        "--dendrite",
        help="Remove dendrites",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--tophat",
        help="Apply tophat filter",
        action="store_true",
        default=False,
    )
    return parser


def get_directories(
    graphical: bool, input_dir: str, output_dir: str
) -> tuple[Path, Path]:
    """Get input and output directories.

    Args:
        graphical: Whether to use GUI.
        input_dir: Input directory path.
        output_dir: Output directory path.

    Returns:
        Tuple of (input_path, output_path).
    """
    if graphical:
        root = tk.Tk()
        root.withdraw()
        input_dir = filedialog.askdirectory(title="Select input directory")
        output_dir = filedialog.askdirectory(title="Select output directory")

    return Path(input_dir), Path(output_dir)


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    input_path, output_path = get_directories(
        args.graphical,
        args.input,
        args.output,
    )

    if not input_path.exists():
        print("Input directory does not exist", flush=True)
        return

    processor = StackProcessor(
        apply_tophat=args.tophat,
        remove_dendrites=args.dendrite,
    )

    files = sorted(input_path.glob("*.tif*"))
    if not files:
        print("No files found in input directory", flush=True)
        return

    print(len(files), flush=True)  # Progress indicator
    for file in files:
        try:
            processed = processor.process(file)
            output_file = output_path / f"{file.stem}.tif"
            cv2.imwrite(str(output_file), processed)
            print(f"Processed {file.name}", flush=True)
        except Exception as e:
            print(f"Failed to process {file.name}: {str(e)}", flush=True)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
