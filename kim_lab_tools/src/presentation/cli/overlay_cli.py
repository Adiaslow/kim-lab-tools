"""CLI for prediction overlay visualization."""

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from ..utils.visualization import save_prediction_overlays


def get_directories() -> tuple[Path, Path]:
    """Get input/output directories from user.

    Returns:
        Tuple of (input_dir, output_dir).
    """
    root = tk.Tk()
    root.withdraw()

    input_dir = filedialog.askdirectory(title="Select input directory")
    if not input_dir:
        raise ValueError("No input directory selected")

    output_dir = filedialog.askdirectory(title="Select output directory")
    if not output_dir:
        raise ValueError("No output directory selected")

    return Path(input_dir), Path(output_dir)


def main() -> None:
    """Main entry point."""
    try:
        input_dir, output_dir = get_directories()
        save_prediction_overlays(input_dir, output_dir)
        print("Successfully created prediction overlays!")
    except Exception as e:
        print(f"Failed to create overlays: {str(e)}")


if __name__ == "__main__":
    main()
