"""CLI for atlas visualization."""

from pathlib import Path

from ..processors.atlas_processor import AtlasProcessor


def main() -> None:
    """Main entry point."""
    # Default regions to map
    regions_to_map = [
        "VISp",
        "VISl",
        "VISpm",
        "VISam",
        "VISrl",
        "VISal",
        "VISa",
        "VISli",
        "RSPv",
        "TEa",
        "VISpor",
    ]

    # Initialize processor
    processor = AtlasProcessor(
        regions_to_map=regions_to_map,
        class_map_path=Path("~/.belljar/csv/class_map.pkl"),
        atlas_path=Path("~/.belljar/nrrd/annotation_10_all.nrrd"),
    )

    # Create visualization
    dorsal_view = processor.create_dorsal_view()

    # Display result
    cv2.imwrite("dorsal_slice_colored.png", dorsal_view)
    cv2.imshow("Dorsal View", dorsal_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
