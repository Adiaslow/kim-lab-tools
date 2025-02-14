"""CLI for region counting."""

import argparse
import csv
from pathlib import Path
import pickle

from ..processors.region_counter import RegionCounter


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Count objects in brain regions")
    parser.add_argument("-o", "--output", help="Output directory", required=True)
    parser.add_argument(
        "-p", "--predictions", help="Predictions directory", required=True
    )
    parser.add_argument(
        "-a", "--annotations", help="Annotations directory", required=True
    )
    parser.add_argument("-m", "--structures", help="Structure map file", required=True)
    parser.add_argument(
        "-l", "--layers", help="Count layers separately", action="store_true"
    )
    return parser


def write_results(
    output_path: Path,
    counts: dict,
    colocalization: dict,
    structures: dict,
) -> None:
    """Write counting results to CSV.

    Args:
        output_path: Output file path.
        counts: Counting results dictionary.
        colocalization: Colocalization results dictionary.
        structures: Structure mapping dictionary.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write per-section counts
        for section, section_counts in counts.items():
            writer.writerow([section])
            writer.writerow(
                ["Region", "Area"] + [f"Channel {c}" for c in section_counts]
            )

            for region in sorted(section_counts[0].keys()):
                row = [region, structures[region]["name"]]
                for channel in section_counts:
                    row.append(section_counts[channel].get(region, 0))
                writer.writerow(row)
            writer.writerow([])

        # Write colocalization
        writer.writerow(["Colocalization"])
        for section, section_coloc in colocalization.items():
            writer.writerow([section])
            for c1, channel_coloc in section_coloc.items():
                row = [f"Channel {c1}"]
                for c2 in sorted(channel_coloc.keys()):
                    row.append(channel_coloc[c2])
                writer.writerow(row)
            writer.writerow([])


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    print(2, flush=True)

    # Setup paths
    pred_dir = Path(args.predictions.strip())
    anno_dir = Path(args.annotations.strip())
    struct_file = Path(args.structures.strip())
    output_dir = Path(args.output.strip())
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process counts
    counter = RegionCounter(include_layers=args.layers)
    counts, colocalization = counter.process((pred_dir, anno_dir, struct_file))

    # Load structures for output
    with open(struct_file, "rb") as f:
        structures = pickle.load(f)

    # Write results
    write_results(
        output_dir / "count_results.csv",
        counts,
        colocalization,
        structures,
    )

    print("Done!", flush=True)


if __name__ == "__main__":
    main()
