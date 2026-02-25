#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dotenv import load_dotenv
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchinfo import summary
import wandb

load_dotenv()

from src.datasets.huggingfacefilelistdataset import HuggingFaceFileListDataset
from src.models.convoluationaltransformer.convtrans_v1 import Captcha_Convolutional_Transformer_V1

wandb.init(
    project="Captcha-Classifier",
    name="Convolutional-Transformer:v3",
    config={
        "learning_rate": 0.0004,
        "architecture": "ConvTrans_V1",
        "dataset": "clean_images_v2",
        "epochs": 50,
        "batch_size": 128,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau"
    }
)


# In[2]:


captcha_transformation = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((40, 150)),
        transforms.RandomInvert(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]
)
dataset = HuggingFaceFileListDataset(
    file_list="/nfs/home/tpz8688/Captcha-Recognition/notebooks/data-cleaning/clean_images_v2.txt", 
    transform=captcha_transformation
)


# In[3]:


train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


# In[4]:


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


# In[5]:


train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=captcha_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=captcha_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=captcha_collate_fn)

# In[47]:


test_indices = test_dataset.indices
test_paths = [str(dataset.image_paths[i]) for i in test_indices]

with open("test_file_list_convtrans_v3.txt", "w") as f:
    for path in test_paths:
        f.write(path + "\n")

print(f"Saved {len(test_paths)} pointers to test_file_list_convtrans_v3.txt")

# In[6]:


model = Captcha_Convolutional_Transformer_V1(num_chars=len(dataset.vocab) + 1)  # Add Blank Character to Vocabulary


# In[7]:


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0

    # Added enumerate to get 'i' for batch-level logging
    for i, (images, labels, lengths) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} Training")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images) # [Batch, Time, Vocab]

        # Prepare for CTC (Log Softmax + Permute)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)

        batch_size = images.size(0)
        input_lengths = torch.full(
            (batch_size,), log_probs.size(0), dtype=torch.long
        )
        target_lengths = lengths 

        # Calculate Loss
        loss = criterion(log_probs, labels, input_lengths, target_lengths)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Log batch loss every 50 steps to monitor transformer stability
        if i % 50 == 0:
            wandb.log({"batch/train_loss": loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss

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
            input_lengths = torch.full(
                (batch_size,), log_probs.size(0), dtype=torch.long
            )
            target_lengths = lengths 

            loss = criterion(log_probs, labels, input_lengths, target_lengths)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss


# In[ ]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Log gradients and transformer topology (log_freq increased for large dataset)
wandb.watch(model, log="all", log_freq=1000)

criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

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

    # 2. Log epoch-level metrics
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
        wandb.run.summary["best_val_loss"] = best_val_loss
        
        model.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), model.SAVE_DIR / "v3.pth")
        print(f"--> New best model saved")

    print("-" * 30)

# 3. Finalize run
wandb.finish()
