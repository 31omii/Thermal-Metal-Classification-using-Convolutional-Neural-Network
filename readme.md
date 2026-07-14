# Thermal Image-Based Metal Classification Using Deep Learning

A deep learning pipeline for classifying six metals from thermal camera sequences captured during controlled cooling experiments using CNNs and sequence models.

---

## Overview

Metals are heated to ~220°C and cooled on a black surface. A thermal camera records the cooling process. Because each metal has a unique thermal decay constant (Newton's Law of Cooling), the cooling curve encodes discriminative information for classification.

**Metals classified:** Iron, Aluminium, Copper, Brass, Zinc, Lead

---

## Dataset

| Property | Details |
|---|---|
| Total frames | 25,344 |
| Frames per class | 4,224 |
| Experiments per metal | 4 (exp1–exp4) |
| Camera | Fluke RSE30H thermal camera |
| Split strategy | Experiment-level (prevents temporal leakage) |

**Split:**
- Train: exp1 + exp2
- Val: exp3
- Test: exp4

Frame-level random splits are not used. Adjacent frames within a single cooling sequence differ by less than 0.1°C, so any frame-level split causes data leakage and inflates validation accuracy.

---

## Project Structure

```
project/
├── data/
│   ├── train/
│   │   ├── Iron/
│   │   ├── Aluminium/
│   │   ├── Copper/
│   │   ├── Brass/
│   │   ├── Zinc/
│   │   └── Lead/
│   ├── val/
│   └── test/
├── models/          # Saved model checkpoints
├── results/         # Accuracy, F1, confusion matrices
├── dataset.py       # Dataset and DataLoader definitions
├── train.py         # Training loop with logging
├── evaluate.py      # Evaluation and metrics
├── predict.py       # Inference on new sequences
└── requirements.txt
```

---

## Models

Eleven architectures are evaluated:

| Type | Models |
|---|---|
| Custom CNN | CustomCNN |
| Transfer learning | ResNet18, EfficientNet-B0, DenseNet121, ConvNeXt-Tiny |
| Attention-based | ViT-B/16, Swin-T |
| Sequence models | CNN+LSTM, CNN+GRU |
| Lightweight | MobileNetV3, ShuffleNetV2 |

CNN+LSTM is the physically motivated baseline — thermal decay follows a temporal curve, so modelling the sequence explicitly aligns with the underlying physics.

---

## Setup

```bash
# Clone and enter the project
git clone <repo-url>
cd <project-dir>

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

**Train a model:**
```bash
python train.py --model resnet18 --epochs 50 --batch_size 32
```

**Evaluate on the test set:**
```bash
python evaluate.py --model resnet18 --checkpoint models/resnet18_best.pth
```

**Run inference:**
```bash
python predict.py --input path/to/thermal_sequence/ --model resnet18
```

---

## Hardware

| Device | VRAM | Role |
|---|---|---|
| NVIDIA GeForce GTX 1650 Ti | 4 GB | Local development |
| NVIDIA RTX A4000 | 16 GB | Full training runs |

---

## Stack

- Python 3.10
- PyTorch
- NumPy
- scikit-learn (metrics)
- Fluke RSE30H (data acquisition)

---

## Key Design Decisions

- **Experiment-level splitting** is mandatory to avoid temporal leakage across cooling sequences.
- **CNN+LSTM** is the primary architecture because the thermal decay curve is a real temporal signal grounded in physics, not an artifact of data ordering.
- **Transfer learning** baselines use ImageNet-pretrained weights adapted to single-channel thermal input.
