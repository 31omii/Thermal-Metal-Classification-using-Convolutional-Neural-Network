import time
import torch.optim as optim
import random
import numpy as np
import argparse

from config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    MODEL_NAME,
    DATA_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    NUM_CLASSES,
    MODEL,
    OUTPUT_DIR,
    RANDOM_SEED,
)

from torch.utils.data import DataLoader

from dataset import ThermalDataset
from transforms import train_transform, val_transform
from model import get_model

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Thermal Metal Classification Benchmark"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "custom_cnn",
            "resnet18",
            "resnet50",
            "densenet121",
            "efficientnet_b0",
            "efficientnet_b3",
        ],
        help="Train a single model",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all models sequentially",
    )

    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):
            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = 100.0 * correct / total

    epoch_time = time.time() - start_time

    return epoch_loss, epoch_accuracy, epoch_time


def validate(model, dataloader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type="cuda"):
                outputs = model(images)

                loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)

    epoch_accuracy = 100.0 * correct / total

    epoch_time = time.time() - start_time

    return epoch_loss, epoch_accuracy, epoch_time


def train_model(model, model_name, train_loader, val_loader):

    output_dir = OUTPUT_DIR / model_name

    model_dir = output_dir / "model"

    plot_dir = output_dir / "plots"

    report_dir = output_dir / "reports"

    log_dir = output_dir / "logs"

    for folder in (
        model_dir,
        plot_dir,
        report_dir,
        log_dir,
    ):
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE[model_name], weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    scaler = GradScaler("cuda")

    best_val_acc = 0.0

    patience_counter = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"\nTraining : {model.__class__.__name__}\n")

    print("-" * 110)
    print(
        f"{'Epoch':<8}"
        f"{'Train Loss':<15}"
        f"{'Train Acc':<12}"
        f"{'Val Loss':<15}"
        f"{'Val Acc':<12}"
        f"{'LR':<12}"
        f"{'Time(s)':<10}"
        f"{'Status'}"
    )
    print("-" * 110)

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_time = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )

        val_loss, val_acc, val_time = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        status = ""

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                model_dir / MODEL_NAME,
            )

            status = "BEST"

        else:
            patience_counter += 1

        print(
            f"{epoch + 1:<8}"
            f"{train_loss:<15.4f}"
            f"{train_acc:<12.2f}"
            f"{val_loss:<15.4f}"
            f"{val_acc:<12.2f}"
            f"{current_lr:<12.2e}"
            f"{train_time:<10.2f}"
            f"{status}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")

            break

    return history


if __name__ == "__main__":
    args = parse_arguments()

    set_seed(RANDOM_SEED)

    # Training dataset
    train_dataset = ThermalDataset(DATA_DIR / "train", transform=train_transform)

    # Validation dataset
    val_dataset = ThermalDataset(DATA_DIR / "val", transform=val_transform)

    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    # Validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print("\nThermal Metal Classification")
    print("-" * 40)

    print(f"Device      : {DEVICE}")
    print(f"Random Seed : {RANDOM_SEED}")
    print(f"Train Images: {len(train_dataset)}")
    print(f"Val Images  : {len(val_dataset)}")
    print(f"Classes     : {NUM_CLASSES}")
    print(f"Batch Size  : {BATCH_SIZE}")
    print(f"Epochs      : {EPOCHS}")

    if args.all:
        models_to_train = MODEL

    elif args.model:
        models_to_train = [args.model]

    else:
        raise ValueError("Specify either --model <model_name> or --all")

    for model_name in models_to_train:
        print("\n")
        print("=" * 80)
        print(f"Training {model_name}")
        print("=" * 80)

        model = get_model(model_name)

        history = train_model(
            model,
            model_name,
            train_loader,
            val_loader,
        )

    print("\nTraining Completed.")