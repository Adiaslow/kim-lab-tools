"""Command line interface for intensity analysis."""

import argparse
import pickle
from pathlib import Path
import os
from typing import Dict, Any

from ..loaders.image_loader import ImageLoader
from ..reconstructors.region_reconstructor import RegionReconstructor
from ..analyzers.intensity_analyzer import IntensityAnalyzer
from demons import resize_image_nearest_neighbor


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the intensity analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Calculate average intensity of regions in normalized coordinates"
    )
    parser.add_argument(
        "-i", "--images", help="input directory for intensity images", default=""
    )
    parser.add_argument(
        "-o", "--output", help="output directory for average intensity pkl", default=""
    )
    parser.add_argument(
        "-a", "--annotations", help="input directory for annotation pkls", default=""
    )
    parser.add_argument(
        "-m", "--map", help="input directory for structure map", default=""
    )
    parser.add_argument(
        "-w",
        "--whole",
        help="Set True to process a whole brain slice (Default is False)",
        default=False,
    )
    return parser


def process_images(args: argparse.Namespace) -> None:
    """Process images according to CLI arguments.

    Args:
        args: Parsed command line arguments.
    """
    # Initialize processors
    image_processor = ImageLoader()
    region_processor = RegionProcessor()

    # Load structure map
    with open(args.map.strip(), "rb") as f:
        structure_map = pickle.load(f)

    intensity_analyzer = IntensityAnalyzer(
        structure_map=structure_map, whole_brain=eval(args.whole.strip())
    )

    # Process each image
    intensity_files = sorted(os.listdir(args.images.strip()))
    annotation_files = sorted(
        f for f in os.listdir(args.annotations.strip()) if f.endswith(".pkl")
    )

    for i, (img_name, ann_name) in enumerate(zip(intensity_files, annotation_files)):
        try:
            # Load and process image
            img_path = Path(args.images.strip()) / img_name
            intensity = image_processor.process(img_path)
            height, width = intensity.shape

            # Load and process annotation
            ann_path = Path(args.annotations.strip()) / ann_name
            with open(ann_path, "rb") as f:
                annotation = pickle.load(f)

            # Resize annotation to match intensity image
            annotation_rescaled = resize_image_nearest_neighbor(
                annotation, (width, height)
            )

            # Process intensities
            intensities = intensity_analyzer.process((intensity, annotation_rescaled))

            # Save results
            save_results(
                intensities=intensities,
                structure_map=structure_map,
                img_name=img_name,
                output_dir=args.output.strip(),
            )

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}", flush=True)
            continue


def save_results(
    intensities: Dict[int, Dict],
    structure_map: Dict[int, Dict[str, Any]],
    img_name: str,
    output_dir: str,
) -> None:
    """Save processing results to files.

    Args:
        intensities: Dictionary of processed intensities by region.
        structure_map: Dictionary mapping structure IDs to metadata.
        img_name: Name of the processed image file.
        output_dir: Output directory path.
    """
    name = Path(img_name).stem

    for region_id, region_data in intensities.items():
        if not region_data:  # Skip empty regions
            continue

        region_name = structure_map[region_id]["acronym"]
        output_path = Path(output_dir) / f"{name}_{region_name}.pkl"

        with open(output_path, "wb") as f:
            pickle.dump(
                {
                    "roi": region_data,
                    "name": region_name,
                },
                f,
            )


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    process_images(args)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
