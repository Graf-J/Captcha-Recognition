#!/usr/bin/env python
# coding: utf-8

# In[15]:


from dotenv import load_dotenv
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import v2
import wandb

load_dotenv()

from src.datasets.generateddataset import GeneratedDataset
from src.datasets.huggingfacefilelistdataset import HuggingFaceFileListDataset
from src.models.crnn.crnn_v1 import Captcha_CRNN_V1
from src.transformation.randomelastictransform import RandomElasticTransform


# In[16]:


wandb.init(
    project="Captcha-Classifier",
    name="CRNN:v5",
    config={
        "learning_rate": 0.0002,
        "architecture": "CRNN_V1",
        "dataset": "fine-tune-combined-images-augmented",
        "epochs": 50,
        "batch_size": 128,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau"
    }
)


# In[17]:


captcha_transformation = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((40, 150)),
    transforms.ToTensor()
])

generated_dataset = GeneratedDataset(
    root_dir="/nfs/home/tpz8688/Captcha-Recognition/data/generated", 
    transform=captcha_transformation, 
    preload=False
)
hf_dataset = HuggingFaceFileListDataset(
    file_list="/nfs/home/tpz8688/Captcha-Recognition/notebooks/data-cleaning/clean_images_v2.txt", 
    transform=captcha_transformation
)


# In[18]:


num_samples = min(50000, len(hf_dataset))
indices = random.sample(range(len(hf_dataset)), num_samples)
hf_subset = Subset(hf_dataset, indices)


# In[19]:


combined_dataset = ConcatDataset([hf_subset, generated_dataset])


# In[20]:


print(f"HuggingFace Subset: {len(hf_subset)} images")
print(f"Generated Dataset: {len(generated_dataset)} images")
print(f"Total Combined Training Data: {len(combined_dataset)} images")


# In[21]:


train_size = int(0.8 * len(combined_dataset))
val_size = int(0.1 * len(combined_dataset))
test_size = len(combined_dataset) - train_size - val_size


# In[22]:


def captcha_collate_fn(batch):
    """
    batch is a list of tuples: [(image, label, length), ...]
    """
    images, labels, lengths = zip(*batch)

    # Stack images into [Batch, Channels, H, W]
    images = torch.stack(images, 0)

    # Pad labels into [Batch, Max_Label_Len_In_Batch]
    # padding_value=0 is the 'blank' index
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    # Convert lengths to a tensor
    label_lengths = torch.tensor(lengths, dtype=torch.long)

    return images, labels_padded, label_lengths


# In[23]:


train_dataset, val_dataset, test_dataset = random_split(
    combined_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=captcha_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=captcha_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=captcha_collate_fn)


# In[ ]:


test_indices = test_dataset.indices
test_paths = []

for i in test_indices:
    # combined_dataset.cumulative_sizes helps locate which sub-dataset the index belongs to
    dataset_idx = 0
    while i >= combined_dataset.cumulative_sizes[dataset_idx]:
        dataset_idx += 1
    
    # Adjust index for the specific sub-dataset
    if dataset_idx > 0:
        local_idx = i - combined_dataset.cumulative_sizes[dataset_idx - 1]
    else:
        local_idx = i
        
    # Get the actual dataset object (could be a Subset or a full Dataset)
    target_ds = combined_dataset.datasets[dataset_idx]
    
    # Handle Subsets vs straight Datasets
    if isinstance(target_ds, torch.utils.data.Subset):
        # Map subset index back to original dataset index
        actual_path = target_ds.dataset.image_paths[target_ds.indices[local_idx]]
    else:
        actual_path = target_ds.image_paths[local_idx]
        
    test_paths.append(str(actual_path))

# Save the file
with open("test_file_list_crnn_v5.txt", "w") as f:
    for path in test_paths:
        f.write(path + "\n")

print(f"Saved {len(test_paths)} pointers to test_file_list_crnn_v5.txt")


# In[24]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Captcha_CRNN_V1(num_chars=len(generated_dataset.vocab) + 1)  # Add Blank Character to Vocabulary
# state_dict = torch.load(
#     Captcha_CRNN_V1.SAVE_DIR / "v3.pth", 
#     map_location=device,
#     weights_only=True 
# )
# model.load_state_dict(state_dict)
# model.to(device)
# model.train()
# print("Successfully loaded v3 weights for fine-tuning!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Captcha_CRNN_V1(num_chars=len(generated_dataset.vocab) + 1)  # Add Blank Character to Vocabulary
state_dict = torch.load(
    Captcha_CRNN_V1.SAVE_DIR / "v5.pth", 
    map_location=device,
    weights_only=True 
)
model.load_state_dict(state_dict)
model.to(device)
model.train()
print("Successfully loaded v5 weights for fine-tuning!")

# In[25]:


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    # Added enumerate to get 'i' for batch-level logging
    for i, (images, labels, lengths) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        images = images.to(device)
        labels = labels.to(device)

        # --- GPU AUGMENTATION START ---
        # v2.ElasticTransform can take a batch of tensors and process them in one go
        # We instantiate it once or use the functional v2
        with torch.no_grad():
            # Randomize alpha for variety
            alpha = float(random.uniform(0.0, 100.0))
            
            # v2 functional is much more flexible with keyword names
            # If you are on an older torchvision, we use the v2 object instead:
            augmentor = v2.ElasticTransform(alpha=alpha, sigma=9.0, fill=1.0).to(device)
            images = augmentor(images)
            
            # Add rotation/affine on GPU
            images = v2.functional.affine(
                images, angle=random.uniform(-8, 8), 
                translate=[0,0], scale=1.0, shear=[0,0], fill=1.0
            )
        # --- GPU AUGMENTATION END ---

        optimizer.zero_grad()

        # Forward pass
        logits = model(images) 
        log_probs = torch.nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)

        batch_size = images.size(0)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
        target_lengths = lengths 

        # Calculate Loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log batch loss every 50 steps to see "noisy" real-time progress
        if i % 50 == 0:
            wandb.log({"batch/train_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# In[26]:


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels, lengths in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)

            batch_size = images.size(0)
            input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
            target_lengths = lengths 

            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss


# In[ ]:



# Log gradients and model topology
wandb.watch(model, log="all", log_freq=1000)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate) # Use config!

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5,
    patience=3,
)

num_epochs = wandb.config.epochs
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # 1. Train and Validate
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
    val_loss = validate(model, val_loader, criterion, device)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # 2. Log everything to WandB for this epoch
    wandb.log({
        "epoch/epoch": epoch + 1,
        "epoch/train_loss": train_loss,
        "epoch/val_loss": val_loss,
        "epoch/learning_rate": current_lr
    })

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Mark the best in WandB Summary
        wandb.run.summary["best_val_loss"] = best_val_loss

        model.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model.SAVE_DIR / "v5.pth")
        print(f"--> New best model saved")

    print("-" * 30)

# 3. Close the WandB run cleanly
wandb.finish()

