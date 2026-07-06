import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import get_model
from transforms import test_transform


CLASS_NAMES = config.CLASS_NAMES


def load_model(model_name, device):

    checkpoint = (
        config.OUTPUT_DIR
        / model_name
        / "model"
        / config.MODEL_NAME
    )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"\nCheckpoint not found:\n{checkpoint}"
        )

    model = get_model(model_name)

    checkpoint = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)

    model.eval()

    return model

def predict_roi(model, roi, device):

    # OpenCV uses BGR, convert to RGB
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL Image
    roi = Image.fromarray(roi)

    # Apply test transforms
    tensor = test_transform(roi)

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    # Move to GPU/CPU
    tensor = tensor.to(device)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = F.softmax(outputs, dim=1)

    probabilities = probabilities.squeeze(0).cpu()

    confidence, prediction = torch.max(
        probabilities,
        dim=0,
    )

    prediction = prediction.item()

    confidence = confidence.item() * 100

    return (
        CLASS_NAMES[prediction],
        confidence,
        probabilities.numpy(),
    )

def process_image(model, image_path, device):

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image:\n{image_path}"
        )

    output = image.copy()

    print()
    print("=" * 60)
    print("Select ROI for each metal cube")
    print("Press ENTER after selecting each ROI")
    print("=" * 60)

    predictions = []

    for i in range(6):

        print(f"\nSelect Cube {i+1}")

        roi = cv2.selectROI(
            "Select Cube",
            image,
            showCrosshair=True,
            fromCenter=False,
        )

        cv2.destroyWindow("Select Cube")

        x, y, w, h = map(int, roi)

        if w == 0 or h == 0:
            print("Skipped.")
            continue

        crop = image[
            y:y+h,
            x:x+w,
        ]

        label, confidence, probabilities = predict_roi(
            model,
            crop,
            device,
        )

        predictions.append(
            {
                "cube": i + 1,
                "label": label,
                "confidence": confidence,
            }
        )

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            output,
            f"{label} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    print()
    print("=" * 60)
    print("Predictions")
    print("=" * 60)

    for result in predictions:

        print(
            f"Cube {result['cube']:<2}"
            f" -> "
            f"{result['label']:<12}"
            f"{result['confidence']:.2f}%"
        )

    return output

def main():

    parser = argparse.ArgumentParser(
        description="Thermal Metal Classification Prediction"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "custom_cnn",
            "resnet18",
            "resnet50",
            "densenet121",
            "efficientnet_b0",
            "efficientnet_b3",
        ],
        help="Model to use",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to thermal image",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nLoading model...")

    model = load_model(
        args.model,
        device,
    )

    print("Model Loaded Successfully.")

    image_path = Path(args.image)

    result = process_image(
        model,
        image_path,
        device,
    )

    prediction_dir = config.OUTPUT_DIR / "predictions"

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = prediction_dir / f"{args.model}_{image_path.stem}.png"

    cv2.imwrite(
        str(save_path),
        result,
    )

    print(f"\nPrediction image saved to:\n{save_path}")

    cv2.imshow(
        "Prediction",
        result,
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()