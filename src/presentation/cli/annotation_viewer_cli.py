"""CLI for annotation viewer."""

import sys
import argparse
import pickle
from pathlib import Path
from qtpy.QtWidgets import QApplication

from ..viewers.annotation_viewer import AnnotationViewer


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Allow adjustment of region alignments"
    )
    parser.add_argument(
        "-a", "--annotations", help="annotation files path", required=True
    )
    parser.add_argument("-i", "--images", help="images path", required=True)
    parser.add_argument("-s", "--structures", help="structures map", required=True)
    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    print(2, flush=True)
    print("Viewing...", flush=True)

    # Load data
    images_path = Path(args.images.strip())
    annotations_path = Path(args.annotations.strip())
    structure_map_path = Path(args.structures.strip())

    with open(structure_map_path, "rb") as f:
        structure_map = pickle.load(f)

    # Create application
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(lambda: print("Done!", flush=True))

    # Create and show viewer
    window = AnnotationViewer(images_path, annotations_path, structure_map)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
