import string
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as ImageType
from pathlib import Path
from tqdm import tqdm

class CachableCaptchaDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, preload: bool = True) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.jpg"))
        self.preload = preload

        # Setup Vocabulary
        self.vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.vocab)}
        self.idx_to_char[0] = ""

        # Caching Containers
        self.cached_images = []
        self.cached_labels = []

        if self.preload:
            print(f"Caching {len(self.image_paths)} images to RAM...")
            for img_path in tqdm(self.image_paths):
                # Load image into RAM
                image = Image.open(img_path).convert("RGB")
                self.cached_images.append(image)

                # Pre-encode label to save CPU time during training
                label_str = img_path.stem
                label_encoded = [
                    self.char_to_idx[char] for char in label_str if char in self.char_to_idx
                ]
                self.cached_labels.append(torch.tensor(label_encoded, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[ImageType, torch.Tensor]:
        if self.preload:
            # High speed: Pull from RAM
            image = self.cached_images[idx]
            label_tensor = self.cached_labels[idx]
        else:
            # Fallback: Pull from Disk
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            label_str = img_path.stem
            label_encoded = [
                self.char_to_idx[char] for char in label_str if char in self.char_to_idx
            ]
            label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        # Transforms stay here so RandomInvert still varies every epoch
        if self.transform:
            image = self.transform(image)

        return image, label_tensor