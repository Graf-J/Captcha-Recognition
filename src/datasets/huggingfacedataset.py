import string
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class HuggingFaceDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, preload: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.jpg"))
        self.preload = preload

        # Setup Vocabulary
        self.vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.vocab)}
        self.idx_to_char[0] = ""

        self.cached_images = []
        self.cached_labels = []

        if self.preload:
            # WARNING: Only use this if you have ~128GB+ of RAM for 1M+ images
            print(f"Caching {len(self.image_paths)} images to RAM...")
            for img_path in tqdm(self.image_paths):
                image = Image.open(img_path).convert("RGB")
                self.cached_images.append(image)

                # FIX: Split at the last underscore to remove the index
                label_str = img_path.stem.split('_')[0]
                label_encoded = [self.char_to_idx[char] for char in label_str if char in self.char_to_idx]
                self.cached_labels.append(torch.tensor(label_encoded, dtype=torch.long))
    
    @property
    def labels(self) -> list[torch.Tensor]:
        if self.preload:
            return self.cached_labels
        else:
            labels = []
            for p in self.image_paths:
                # Split at the last underscore (remove index)
                label_str = p.stem.split("_")[0]

                label_encoded = [
                    self.char_to_idx[c]
                    for c in label_str
                    if c in self.char_to_idx
                ]

                labels.append(torch.tensor(label_encoded, dtype=torch.long))

            return labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self.preload:
            image = self.cached_images[idx]
            label_tensor = self.cached_labels[idx]
        else:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            
            # FIX: Split at the last underscore to remove the index
            label_str = img_path.stem.split('_')[0]
            
            label_encoded = [
                self.char_to_idx[char] for char in label_str if char in self.char_to_idx
            ]
            label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        # Return the label and its actual length (essential for CTC Loss)
        return image, label_tensor, len(label_tensor)