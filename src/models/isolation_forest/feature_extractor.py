import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm


class CaptchaFeatureExtractor:
    """Wraps one of the four pre-trained CAPTCHA CNN backbones as a frozen feature extractor.

    Loads the model from HuggingFace, extracts the shared CNN backbone, and exposes
    an extract() method that maps a DataLoader of images to a (N, 256) feature matrix
    via global average pooling.

    Supported model names:
        "crnn_base"            -> Graf-J/captcha-crnn-base
        "crnn_finetuned"       -> Graf-J/captcha-crnn-finetuned
        "convtrans_base"       -> Graf-J/captcha-conv-transformer-base
        "convtrans_finetuned"  -> Graf-J/captcha-conv-transformer-finetuned
    """

    MODEL_IDS = {
        "crnn_base":           "Graf-J/captcha-crnn-base",
        "crnn_finetuned":      "Graf-J/captcha-crnn-finetuned",
        "convtrans_base":      "Graf-J/captcha-conv-transformer-base",
        "convtrans_finetuned": "Graf-J/captcha-conv-transformer-finetuned",
    }

    def __init__(self, model_name: str, device: torch.device | None = None) -> None:
        if model_name not in self.MODEL_IDS:
            raise ValueError(
                f"model_name must be one of {list(self.MODEL_IDS.keys())}, got '{model_name}'"
            )

        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading {self.MODEL_IDS[model_name]} on {self.device}...")
        model = AutoModel.from_pretrained(self.MODEL_IDS[model_name], trust_remote_code=True)

        # The CRNN uses `conv_layer`; the ConvTransformer uses `conv`
        if hasattr(model, "conv_layer"):
            self.backbone = model.conv_layer
        elif hasattr(model, "conv"):
            self.backbone = model.conv
        else:
            raise AttributeError(
                f"Could not find CNN backbone on {self.MODEL_IDS[model_name]}. "
                "Expected attribute 'conv_layer' (CRNN) or 'conv' (ConvTransformer)."
            )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()
        self.backbone.to(self.device)

    def extract(self, dataloader: DataLoader) -> np.ndarray:
        """Run all batches through the CNN backbone and return a (N, 256) feature matrix.

        Args:
            dataloader: yields (images, *) where images is (B, 1, H, W), values in [0, 1]

        Returns:
            np.ndarray of shape (N, 256)
        """
        all_features = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting [{self.model_name}]"):
                images = batch[0].to(self.device)        # (B, 1, H, W)
                features = self.backbone(images)          # (B, 256, H', W')
                features = features.mean(dim=[-2, -1])   # Global avg pool → (B, 256)
                all_features.append(features.cpu().numpy())

        if not all_features:
            raise RuntimeError(
                "DataLoader yielded no batches — the dataset is empty. "
                "Check that the data directory exists and contains images."
            )

        return np.concatenate(all_features, axis=0)
