import os
from pathlib import Path

from dotenv import load_dotenv
from torchvision.datasets import CIFAR10
from tqdm import tqdm

load_dotenv()

output_dir = Path(os.environ["PROJECT_ROOT_DIR"]) / "data" / "cifar10"
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading CIFAR-10 test set...")
dataset = CIFAR10(
    root=Path(os.environ["PROJECT_ROOT_DIR"]) / "data" / "cifar10_raw",
    train=False,
    download=True
)

print(f"Saving {len(dataset)} images to {output_dir}...")
for i, (img, label) in enumerate(tqdm(dataset)):
    img.save(output_dir / f"{i:05d}_label{label}.png")

print(f"Saved {len(dataset)} images to {output_dir}")
