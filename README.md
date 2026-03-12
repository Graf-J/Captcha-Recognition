# Captcha-Recognition

A deep learning system for CAPTCHA OCR, comparing CRNN and Convolutional Transformer architectures. Also includes an Isolation Forest anomaly detector to determine whether a given image is a CAPTCHA.

---

## Models

Four OCR models are trained and published to HuggingFace:

| Model | Architecture | Training Data | HuggingFace |
|-------|-------------|--------------|-------------|
| CRNN Base | CNN + Bi-LSTM | HuggingFace dataset | [Graf-J/captcha-crnn-base](https://huggingface.co/Graf-J/captcha-crnn-base) |
| CRNN Finetuned | CNN + Bi-LSTM | HuggingFace + Generated | [Graf-J/captcha-crnn-finetuned](https://huggingface.co/Graf-J/captcha-crnn-finetuned) |
| Conv-Transformer Base | CNN + Transformer | HuggingFace dataset | [Graf-J/captcha-conv-transformer-base](https://huggingface.co/Graf-J/captcha-conv-transformer-base) |
| Conv-Transformer Finetuned | CNN + Transformer | HuggingFace + Generated | [Graf-J/captcha-conv-transformer-finetuned](https://huggingface.co/Graf-J/captcha-conv-transformer-finetuned) |

An Isolation Forest anomaly detector is also trained on top of the CNN feature extractor from each model to determine whether an input image is a CAPTCHA.

### Isolation Forest Results

Trained on 50,000 CAPTCHA images (subsampled from `hammer888/captcha-data`). Evaluated against 10,000 samples per test set.

| Model | ROC-AUC vs CIFAR-10 | ROC-AUC vs SVHN | FPR@95TPR vs CIFAR-10 | FPR@95TPR vs SVHN |
|-------|:-------------------:|:---------------:|:---------------------:|:-----------------:|
| ConvTransformer Base | **0.9951** | **0.9940** | **0.0062** | **0.0056** |
| ConvTransformer Finetuned | 0.9939 | 0.9917 | 0.0041 | 0.0086 |
| CRNN Base | 0.9834 | 0.9883 | 0.0217 | 0.0110 |
| CRNN Finetuned | 0.9792 | 0.9858 | 0.0172 | 0.0085 |

**Key findings:**
- All four models achieve strong anomaly detection (ROC-AUC > 0.97), confirming the CNN backbones have encoded CAPTCHA-specific structure effectively
- ConvTransformer backbones outperform CRNN across all metrics — self-attention produces richer feature representations than the Bi-LSTM hidden state
- Base models slightly outperform their finetuned counterparts — fine-tuning on generated CAPTCHAs broadens the feature distribution, making it marginally less compact for Isolation Forest
- SVHN (hard negatives) is consistently easier to separate than CIFAR-10 despite visual similarity to CAPTCHAs, likely because 32×32 SVHN images stretched to 40×150 produce feature vectors distant from native-resolution CAPTCHAs
- **Recommended backbone for deployment: ConvTransformer Base** — best ROC-AUC and lowest FPR@95TPR vs SVHN

---

## Data Sources

| Dataset | Purpose | Source |
|---------|---------|--------|
| hammer888/captcha-data | OCR training (base) | HuggingFace |
| akashguna/large-captcha-dataset | Additional CAPTCHA images | Kaggle |
| Python Captcha Library | OCR fine-tuning (generated) | `captcha` pip package |
| CIFAR-10 | Isolation Forest evaluation (easy negatives) | torchvision |
| SVHN (Street View House Numbers) | Isolation Forest evaluation (hard negatives) | torchvision |

---

## Setup

### 1. Install Dependencies

```bash
uv sync
```

On Linux this installs CUDA 13.0 PyTorch. On Windows this installs CUDA 13.0 PyTorch (requires CUDA 12.4+ driver). TensorFlow is Linux-only and skipped on Windows.

### 2. Environment Variables

Create a `.env` file in the repository root:

```
PROJECT_ROOT_DIR=<absolute path to this repository>
```

Example on Windows (use forward slashes to avoid escape character issues):
```
PROJECT_ROOT_DIR=C:/Users/username/Documents/Captcha-Recognition
```

---

## OCR Models

### Step 1 — Download Training Data

```bash
uv run scripts/download-huggingface-dataset.py
uv run scripts/download-kaggle-dataset.py
```

Data lands in `data/hammer_captchas/` and the kagglehub cache respectively.

### Step 2 — Generate Synthetic CAPTCHAs (for fine-tuning)

Run `scripts/generate-dataset.ipynb`. This uses the Python `captcha` library with the Nunito font to produce 200,000 synthetic CAPTCHA images, saved to `data/generated/`.

### Step 3 — Data Cleaning

Run `notebooks/data-cleaning/01_error-analysis.ipynb` to identify and prune mislabeled images from the HuggingFace dataset. Produces `clean_images_v2.txt`.

### Step 4 — Train OCR Models

Training scripts are versioned (v1–v6). Run the final versions for each architecture:

**CRNN:**
```bash
uv run notebooks/crnn/crnn_v4.py   # base model
uv run notebooks/crnn/crnn_v6.py   # fine-tuned (loads v4 weights)
```

**Convolutional Transformer:**
```bash
uv run notebooks/convtrans/convtrans_v4.py   # base model
uv run notebooks/convtrans/convtrans_v6.py   # fine-tuned (loads v4 weights)
```

Weights are saved to `weights/crnn/` and `weights/conv_transformer/`. Training is logged to WandB under the `Captcha-Classifier` project.

### Step 5 — Evaluate OCR Models

Run the notebooks in `notebooks/model-analysis/` in order:

```
01_crnn-analysis.ipynb
02_crnn-hf-evaluation.ipynb
03_crnn-gen-evaluation.ipynb
04_convtrans-analysis.ipynb
05_convtrans-gen-evaluation.ipynb
06_convtrans-hf-evaluation.ipynb
07_crnn-generated-image-inference.ipynb
```

### Step 6 — Deploy to HuggingFace (optional)

Run the deploy notebooks in `scripts/`:

```bash
scripts/deploy_crnn_base.ipynb
scripts/deploy_crnn_finetuned.ipynb
scripts/deploy_convtrans_base.ipynb
scripts/deploy_convtrans_finetuned.ipynb
```

---

## Isolation Forest (CAPTCHA Anomaly Detector)

A one-class anomaly detector trained on CAPTCHA image features. Uses the frozen CNN backbone from each of the four OCR models as a feature extractor, then fits an Isolation Forest on the resulting feature vectors.

### Step 1 — Download Evaluation Datasets

```bash
uv run scripts/download-coco-dataset.py      # CIFAR-10 easy negatives  → data/cifar10/
uv run scripts/download-iiit5k-dataset.py    # SVHN hard negatives       → data/svhn/
```

### Step 2 — Train Isolation Forest Models

Run `notebooks/isolation-forest/isolation_forest_v1.ipynb`.

This extracts CNN features from the CAPTCHA training set using each of the four model backbones and fits one Isolation Forest per model. Fitted models are saved to `weights/isolation_forest/`.

### Step 3 — Evaluate

Run `notebooks/isolation-forest/isolation_forest_evaluation.ipynb`.

Evaluates each of the four Isolation Forest models separately against three test sets:

| Test Set | Type | Result |
|----------|------|--------|
| CIFAR-10 | Easy negatives | ROC-AUC 0.979–0.995 |
| SVHN | Hard negatives | ROC-AUC 0.986–0.994 |

Metrics reported per dataset: ROC-AUC, FPR at 95% TPR.

---

## Autoencoder (CAPTCHA Anomaly Detector)

A autoencoder using Binary Crossentropy Loss is trained on the CAPTCHA images. It is then used to measure the reconstruction error and detect images which do not represent real CAPTCHAs.

### Step 1 — Download Evaluation Datasets

```bash
uv run scripts/download-coco-dataset.py      # CIFAR-10 easy negatives  → data/cifar10/
uv run scripts/download-iiit5k-dataset.py    # SVHN hard negatives       → data/svhn/
```

### Step 2 — Train Autoencoder

Run `notebooks/autoencoder/autoencoder_v1.ipynb`.

### Step 3 — Evaluate

Run `notebooks/autoencoder/autoencoder_model_evaluation.ipynb`.

| Test Set | Type | ROC-AUC | FPR@95TPR
|----------|------|---------|----------
| CIFAR-10 | Easy negatives | 0.9711 | 0.0890
| SVHN | Hard negatives | 0.9711 | 0.1020

---

## Project Structure

```
Captcha-Recognition/
├── .env                          # PROJECT_ROOT_DIR (not tracked)
├── pyproject.toml                # Dependencies (uv)
├── scripts/
│   ├── download-huggingface-dataset.py
│   ├── download-kaggle-dataset.py
│   ├── download-coco-dataset.py  # Downloads CIFAR-10 easy negatives
│   ├── download-iiit5k-dataset.py
│   ├── generate-dataset.ipynb
│   ├── deploy_crnn_base.ipynb
│   ├── deploy_crnn_finetuned.ipynb
│   ├── deploy_convtrans_base.ipynb
│   └── deploy_convtrans_finetuned.ipynb
├── src/
│   ├── datasets/
│   │   ├── kaggledataset.py
│   │   ├── huggingfacedataset.py
│   │   ├── huggingfacefilelistdataset.py
│   │   ├── generateddataset.py
│   │   ├── cifar10dataset.py         # CIFAR-10 easy negatives
│   │   └── svhndataset.py            # SVHN hard negatives
│   ├── models/
│   │   ├── crnn/crnn_v1.py
│   │   ├── convoluationaltransformer/convtrans_v1.py
│   │   └── isolation_forest/
│   │       └── feature_extractor.py
│   └── transformation/
│       └── randomelastictransform.py
├── notebooks/
│   ├── crnn/                     # crnn_v1–v6
│   ├── convtrans/                # convtrans_v1–v6
│   ├── isolation-forest/
│   │   ├── isolation_forest_v1.ipynb
│   │   └── isolation_forest_evaluation.ipynb
│   ├── model-analysis/           # 01–07 evaluation notebooks
│   ├── data-analysis/
│   ├── data-cleaning/
│   └── augmentation/
├── deploy/
│   ├── captcha_crnn_base/
│   ├── captcha_crnn_finetuned/
│   ├── captcha_convolutionaltransformer_base/
│   └── captcha_convolutionaltransformer_finetuned/
├── weights/
│   ├── crnn/                     # v1–v6.pth
│   ├── conv_transformer/         # v1–v6.pth
│   └── isolation_forest/         # v1.joblib (one per model)
└── data/                         # gitignored
    ├── hammer_captchas/
    ├── generated/
    ├── cifar10/
    └── svhn/
```
