import string
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as ImageType
from pathlib import Path


class CaptchaDataset(Dataset):
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.jpg"))

        self.vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.vocab)}
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.vocab)}
        self.idx_to_char[0] = ""  # Blank maps to nothing

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[ImageType, torch.Tensor]:
        # Load Image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Encode Label
        label_str = img_path.stem
        label_encoded = [
            self.char_to_idx[char] for char in label_str if char in self.char_to_idx
        ]
        label_tensor = torch.tensor(label_encoded, dtype=torch.long)

        return image, label_tensor
