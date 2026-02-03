import os
from pathlib import Path
import torch.nn as nn


class Captcha_CRNN_V1(nn.Module):
    SAVE_DIR = Path(os.environ["PROJECT_ROOT_DIR"]) / "weights" / "crnn" / "v1.pth"

    def __init__(self, num_chars):
        super(Captcha_CRNN_V1, self).__init__()
        self.conv_layer = nn.Sequential(
            # Layer 1: 50x140 -> 25x70
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),

            # Layer 2: 25x70 -> 12x35
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),

            # Layer 3: 12x35 -> 6x35 (Pooling only Height)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 1)), # Reduce height to 6, keep width 35

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # New Input Size: Channels (128) * Height (6) = 768
        self.lstm = nn.LSTM(
            input_size=1280, 
            hidden_size=256,
            bidirectional=True, 
            batch_first=True
        )

        self.classifier = nn.Linear(512, num_chars) # 256 * 2 (bidirectional)

    def forward(self, x):
        x = self.conv_layer(x)
        # Reshape for LSTM: (Batch, Channels, Height, Width) -> (Batch, Width, Features)
        x = x.permute(0, 3, 1, 2)
        batch, width, channels, height = x.size()
        x = x.view(batch, width, -1)

        x, _ = self.lstm(x)
        x = self.classifier(x)
        return x
