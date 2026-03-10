import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class SVHNDataset(Dataset):
    """Loads pre-saved SVHN images as hard negatives for Isolation Forest evaluation.

    Images are expected to be saved by scripts/download-iiit5k-dataset.py in the format:
        {index:05d}_label{digit_label}.png

    Returns label=0 for all samples (not a CAPTCHA).
    """

    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(self.root_dir.glob("*.png"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(0)
