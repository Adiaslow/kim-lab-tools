"""DAPI dataset implementation."""

from pathlib import Path
from typing import Optional, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DAPIDataset(Dataset):
    """Dataset for DAPI images."""

    def __init__(
        self,
        image_paths: List[Path],
        transform: Optional[transforms.Compose] = None,
        target_size: tuple[int, int] = (256, 256),
    ):
        """Initialize the dataset.

        Args:
            image_paths: List of paths to images.
            transform: Optional transforms to apply.
            target_size: Size to resize images to.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.target_size = target_size
        self.normalize = transforms.Normalize((20.59695 / 255,), (40.319914 / 255,))

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Get a dataset item.

        Args:
            index: Item index.

        Returns:
            Processed image tensor.
        """
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L")
        image = image.resize(self.target_size, Image.Resampling.BILINEAR)

        if self.transform:
            image = self.transform(image)
            image = self.normalize(image)

        return image
