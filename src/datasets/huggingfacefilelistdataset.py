import string
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class HuggingFaceFileListDataset(Dataset):
    def __init__(self, file_list: str, transform=None):
        self.transform = transform

        with open(file_list, "r") as f:
            self.image_paths = [Path(line.strip()) for line in f if line.strip()]

        # Same vocabulary as your main dataset
        self.vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.vocab)}
        self.idx_to_char[0] = ""

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Label from filename (before underscore)
        label_str = img_path.stem.split("_")[0]
        label_encoded = [
            self.char_to_idx[c]
            for c in label_str
            if c in self.char_to_idx
        ]
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor, len(label_tensor)
