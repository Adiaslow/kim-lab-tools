"""Atlas dataset implementations."""

import cv2
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class AtlasDataset(Dataset):
    """Dataset for atlas images."""

    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose] = None):
        """Initialize dataset.

        Args:
            root_dir: Directory containing images.
            transform: Optional transforms to apply.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(
            [f for f in root_dir.iterdir() if f.is_file() and f.suffix == ".png"]
        )

    def sobel(self, image: np.ndarray) -> np.ndarray:
        """Apply Sobel edge detection.

        Args:
            image: Input image array.

        Returns:
            Edge detected image.
        """
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, delta=25)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, delta=25)

        gx = cv2.convertScaleAbs(gx)
        gy = cv2.convertScaleAbs(gy)

        return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get dataset item.

        Args:
            idx: Item index.

        Returns:
            Processed image tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_files[idx]
        image = Image.open(img_path)
        if image.size != (256, 256):
            image = image.resize((256, 256))
        image = np.array(image)
        image = self.sobel(image)

        if self.transform:
            image = self.transform(image)

        return image
