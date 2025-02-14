"""CLI for image filtering."""

import argparse
import tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path

from ..processors.filter_processor import FilterProcessor


def get_directories(graphical: bool = True) -> tuple[Path, Path]:
    """Get input/output directories.

    Args:
        graphical: Whether to use GUI dialogs.

    Returns:
        Tuple of (input_dir, output_dir).
    """
    if graphical:
        root = tk.Tk()
        root.withdraw()
        input_dir = filedialog.askdirectory(title="Select input directory")
        output_dir = filedialog.askdirectory(title="Select output directory")
    else:
        input_dir = input("Input directory: ")
        output_dir = input("Output directory: ")

    return Path(input_dir), Path(output_dir)


def get_filter_size(graphical: bool = True, default: int = 5) -> int:
    """Get filter size parameter.

    Args:
        graphical: Whether to use GUI dialog.
        default: Default filter size.

    Returns:
        Filter size value.
    """
    if graphical:
        root = tk.Tk()
        root.withdraw()
        size = simpledialog.askinteger(
            title="Filter Size",
            prompt="Enter tophat filter size (pixels):",
            initialvalue=default,
        )
        return size if size else default
    return default


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Apply image filters")
    parser.add_argument("-o", "--output", help="Output directory", default="")
    parser.add_argument("-i", "--input", help="Input directory", default="")
    parser.add_argument("-f", "--filter", help="Tophat filter size", default="")
    parser.add_argument(
        "-c", "--correction", help="Gamma correction value", type=float, default=1.25
    )
    parser.add_argument(
        "-g",
        "--graphical",
        help="Use graphical interface",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    # Get parameters
    input_dir, output_dir = get_directories(args.graphical)
    filter_size = int(args.filter) if args.filter else get_filter_size(args.graphical)

    # Process files
    processor = FilterProcessor(filter_size=filter_size, gamma=args.correction)

    files = sorted(input_dir.glob("*.tif*"))
    print(f"Found {len(files)} files")

    for file in files:
        print(f"Processing {file.name}")
        processor.process_file(file, output_path=output_dir / file.name)

    print("Done!")


if __name__ == "__main__":
    main()
