import os
from pathlib import Path
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, time, d_model)
        return x + self.pe[:, : x.size(1)]


class Captcha_Convolutional_Transformer_V1(nn.Module):
    SAVE_DIR = Path(os.environ["PROJECT_ROOT_DIR"]) / "weights" / "conv_transformer"

    def __init__(
        self,
        num_chars,
        d_model=1280,
        nhead=8,
        num_layers=1,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        # -------------------------
        # CNN feature extractor
        # -------------------------
        self.conv = nn.Sequential(
            # (B, 1, 50, 140) → (B, 32, 25, 70)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),

            # → (B, 64, 12, 35)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),

            # → (B, 128, 6, 35)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            # → (B, 256, 5, 35)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        # 256 * 5 = 1280
        self.feature_dim = d_model

        # -------------------------
        # Positional Encoding
        # -------------------------
        self.positional_encoding = PositionalEncoding(d_model)

        # -------------------------
        # Transformer Encoder
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # more stable in practice
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # -------------------------
        # Classification head
        # -------------------------
        self.classifier = nn.Linear(d_model, num_chars)

    def forward(self, x):
        """
        x: (batch, 1, H, W)
        returns: (batch, time, num_chars)
        """

        # CNN
        x = self.conv(x)
        # (B, C, H, W) = (B, 256, 5, 35)

        # Prepare sequence
        x = x.permute(0, 3, 1, 2)
        # (B, W, C, H) = (B, 35, 256, 5)

        b, t, c, h = x.size()
        x = x.reshape(b, t, c * h)
        # (B, 35, 1280)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer(x)
        # (B, 35, 1280)

        # Per-timestep classification
        x = self.classifier(x)
        # (B, 35, num_chars)

        return x