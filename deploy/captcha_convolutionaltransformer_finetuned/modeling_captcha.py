import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .configuration_captcha import CaptchaConfig

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
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class CaptchaConvolutionalTransformer(PreTrainedModel):
    config_class = CaptchaConfig

    def __init__(self, config):
        super().__init__(config)

        # CNN Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
        )

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(config.d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Classification Head
        self.classifier = nn.Linear(config.d_model, config.num_chars)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, pixel_values, labels=None):
        """
        pixel_values: (batch, 1, H, W)
        """
        # Extract features
        x = self.conv(pixel_values) # (B, 256, H_final, W_final)

        # Prepare sequence: Permute to (Batch, Width, Channels, Height)
        x = x.permute(0, 3, 1, 2)
        b, t, c, h = x.size()
        
        # Flatten Channels and Height into the d_model dimension
        x = x.reshape(b, t, c * h) # (B, T, d_model)

        # Apply Transformer logic
        x = self.positional_encoding(x)
        x = self.transformer(x)
        
        # Map to character logits
        logits = self.classifier(x) # (B, T, num_chars)

        # Return an output object
        return SequenceClassifierOutput(logits=logits)