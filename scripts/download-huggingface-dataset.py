import os
import re
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
load_dotenv()

ds = load_dataset("hammer888/captcha-data")
# output_dir = "./data/hammer_captchas"
output_dir = Path(os.environ["PROJECT_ROOT_DIR"]) / "data" / "hammer_captchas"
os.makedirs(output_dir, exist_ok=True)

print(f"Extracting images from cache to {output_dir}...")

for i, item in enumerate(tqdm(ds['train'])):
    raw_text = item['text'] # The long sentence
    img = item['image']
    
    # 1. Extract ONLY the part inside the single quotes
    # This finds '066615' and extracts just 066615
    match = re.search(r"'(.*?)'", raw_text)
    if match:
        clean_label = match.group(1)
    else:
        # Fallback if the pattern fails for some reason
        clean_label = "unknown"

    # 2. Handle Image Conversion
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 3. Save with the CLEAN label
    img.save(output_dir / f"{clean_label}_{i}.jpg")