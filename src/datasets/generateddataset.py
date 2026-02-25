import string
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class GeneratedDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, preload: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        # Generated data uses .png from the captcha library
        self.image_paths = list(self.root_dir.glob("*.png"))
        self.preload = preload

        # Setup Vocabulary
        self.vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.vocab)}
        self.idx_to_char[0] = ""

        self.cached_images = []
        self.cached_labels = []

        if self.preload:
            print(f"Caching {len(self.image_paths)} generated images to RAM...")
            for img_path in tqdm(self.image_paths):
                image = Image.open(img_path).convert("RGB")
                self.cached_images.append(image)
                
                # Use helper to parse complex filenames
                label_tensor = self._parse_label(img_path)
                self.cached_labels.append(label_tensor)
    
    def _parse_label(self, path: Path) -> torch.Tensor:
        """Extracts text from filenames like 'Ab12_len4_99.png'"""
        # Split by underscore and take the first part
        label_str = path.stem.split('_')[0]
        label_encoded = [
            self.char_to_idx[char] 
            for char in label_str 
            if char in self.char_to_idx
        ]
        return torch.tensor(label_encoded, dtype=torch.long)

    @property
    def labels(self) -> list[torch.Tensor]:
        if self.preload:
            return self.cached_labels
        return [self._parse_label(p) for p in self.image_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self.preload:
            image = self.cached_images[idx]
            label_tensor = self.cached_labels[idx]
        else:
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label_tensor = self._parse_label(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label_tensor, len(label_tensor)