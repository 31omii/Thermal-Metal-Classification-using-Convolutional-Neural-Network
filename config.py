from pathlib import Path
import torch

# Project Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Dataset
CLASS_NAMES = [
    "Aluminium",
    "Brass",
    "Copper",
    "Iron",
    "Lead",
    "Zinc",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 128

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42

# Dataloader
BATCH_SIZE = 32
NUM_WORKERS = 2
PIN_MEMORY = 2

# Training
EPOCHS = 40

LEARNING_RATE = {
    "custom_cnn": 1e-3,
    "resnet18": 1e-4,
    "resnet50": 1e-4,
    "densenet121": 1e-4,
    "efficientnet_b0": 5e-5,
    "efficientnet_b3": 5e-5,
}

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AMP = True

# Model Selection
MODEL = [
    "custom_cnn",
    "resnet18",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
]

# Model Saving
MODEL_NAME = "best_model.pth"