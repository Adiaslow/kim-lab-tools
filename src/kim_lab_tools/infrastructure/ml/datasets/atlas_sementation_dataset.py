"""Dataset implementation for axon segmentation."""

from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from skimage.exposure import equalize_adapthist
from skimage.morphology import binary_dilation, disk
from transformers import SegformerImageProcessor


class AxonSegmentationDataset(Dataset):
    """Dataset for axon segmentation training."""

    # ... implementation ...
