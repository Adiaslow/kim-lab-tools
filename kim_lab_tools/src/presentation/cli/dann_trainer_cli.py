"""CLI for DANN training."""

import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..models.unet import UNetWithGRL
from ..datasets.atlas_dataset import AtlasDataset
from ..datasets.dapi_dataset import DAPIDataset
from ..trainers.dann_trainer import DANNTrainer


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(description="Train DANN for domain adaptation")
    parser.add_argument("--source-dir", help="Source image directory", required=True)
    parser.add_argument("--target-dir", help="Target image directory", required=True)
    parser.add_argument("--map-dir", help="Map directory", required=True)
    parser.add_argument("--weights", help="Path to pretrained weights", required=True)
    parser.add_argument("--output", help="Output directory", required=True)
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size")
    return parser


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup paths
    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    map_dir = Path(args.map_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get file paths
    source_images = sorted(source_dir.glob("*.png"))
    target_images = sorted(target_dir.glob("*.png"))
    map_images = sorted(map_dir.glob("*.tif"))

    # Setup datasets
    transform = transforms.ToTensor()
    source_dataset = AtlasDataset(
        source_images, map_images, transform=transform, random_affine=False
    )
    target_dataset = DAPIDataset(target_images, transform=transform)

    # Setup dataloaders
    source_dataloader = DataLoader(
        source_dataset, batch_size=args.batch_size, shuffle=True
    )
    target_dataloader = DataLoader(
        target_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Setup model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNetWithGRL(1, 1328)
    unet.load_state_dict(torch.load(args.weights))

    trainer = DANNTrainer(unet, num_classes=1328, device=device)
    trainer.train(source_dataloader, target_dataloader, args.epochs, output_dir)


if __name__ == "__main__":
    main()
