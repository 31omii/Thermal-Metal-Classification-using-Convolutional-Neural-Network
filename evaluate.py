import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import argparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from torch.utils.data import DataLoader

from config import (
    DEVICE,
    CLASS_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    MODEL_NAME,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    MODEL
)

from dataset import ThermalDataset
from transforms import test_transform
from model import get_model


def evaluate_model(model_name):

    output_dir = OUTPUT_DIR / model_name

    model_dir = output_dir / "model"

    plot_dir = output_dir / "plots"

    report_dir = output_dir / "reports"

    for folder in (
        model_dir,
        plot_dir,
        report_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    test_dataset = ThermalDataset(
        DATA_DIR / "test",
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers = NUM_WORKERS,
        pin_memory = PIN_MEMORY,
    )

    model = get_model(model_name)

    checkpoint = torch.load(
        model_dir / MODEL_NAME,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    model.eval()

    running_loss = 0.0

    predictions = []

    ground_truth = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE, non_blocking=True)

            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            if len(predictions) < 20:
                for gt, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                    print(
                        f"GT={CLASS_NAMES[gt]:10s}  "
                        f"PRED={CLASS_NAMES[pred]:10s}"
                    )

            predictions.extend(predicted.cpu().numpy())

            ground_truth.extend(labels.cpu().numpy())

    inference_time = (time.time() - start_time) / len(ground_truth)

    test_loss = running_loss / len(test_loader)

    accuracy = accuracy_score(
        ground_truth,
        predictions,
    )

    precision = precision_score(
        ground_truth,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        ground_truth,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        ground_truth,
        predictions,
        average="weighted"
    )

    cm = confusion_matrix(
        ground_truth,
        predictions,
    )

    report_dict = classification_report(
    ground_truth,
    predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        report_dir / "classification_report.csv",
        index=True,
    )

    report_text = classification_report(
        ground_truth,
        predictions,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    with open(report_dir / "classification_report.txt", "w") as f:
        f.write(report_text)

    print(report_text)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(8, 8))

    display.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format="d",
    )

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        plot_dir / "confusion_matrix.png",
        dpi=300,
    )

    plt.close()

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Loss",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "Inference Time (ms/image)",
            ],
            "Value": [
                test_loss,
                accuracy,
                precision,
                recall,
                f1,
                inference_time * 1000,
            ],
        }
    )

    metrics.to_csv(
        report_dir / "metrics.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print(f"Evaluation : {model_name}")
    print("=" * 65)

    print(f"Test Images      : {len(test_dataset)}")
    print(f"Loss             : {test_loss:.4f}")
    print(f"Accuracy         : {accuracy * 100:.2f}%")
    print(f"Precision        : {precision * 100:.2f}%")
    print(f"Recall           : {recall * 100:.2f}%")
    print(f"F1 Score         : {f1 * 100:.2f}%")
    print(f"Inference Time   : {inference_time * 1000:.2f} ms/image")

    print("\nClassification Report\n")

    print("Confusion Matrix      : Saved")
    print("Classification Report : Saved")
    print("Metrics               : Saved")

    return {
        "loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_time": inference_time,
    }

def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Evaluate Thermal Metal Classification Models"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=MODEL,
        help="Evaluate a single model",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all trained models",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    if args.all:

        models_to_evaluate = MODEL

    elif args.model:

        models_to_evaluate = [args.model]

    else:

        raise ValueError(
            "Specify either --model <model_name> or --all"
        )

    for model_name in models_to_evaluate:

        evaluate_model(model_name)
