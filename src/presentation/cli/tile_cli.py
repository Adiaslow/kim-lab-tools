"""CLI for image tiling operations."""

import cv2
import argparse
from pathlib import Path

from ..models.section_image import SectionImage
from ..processors.tile_processor import TileProcessor


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process tissue section images")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--tile-size", type=int, default=303, help="Tile size in pixels"
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Reconstruct image from tiles",
    )
    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    section = SectionImage(args.input, image)

    # Process
    processor = TileProcessor(tile_size=args.tile_size)

    if args.reconstruct:
        reconstructed = processor.reconstruct_from_tiles(
            Path(args.output), section.stem
        )
        output_path = Path(args.output) / f"{section.stem}_reconstructed.png"
        cv2.imwrite(str(output_path), reconstructed)
    else:
        processor.save_tiles(section, args.output)


if __name__ == "__main__":
    main()
