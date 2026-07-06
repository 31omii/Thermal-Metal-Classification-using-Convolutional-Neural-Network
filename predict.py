"""
predict.py
python predict.py --model resnet18 --image D:/metal_clf/test_img.jpg
"""

# D:/metal_clf/dataset/exp1/frame_000006.jpg

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

import config
from model import get_model
from transforms import test_transform

def load_model(model_name: str, checkpoint_path: Path, device: torch.device):
    model = get_model(model_name)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Support different checkpoint formats
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])

    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])

    else:
        # In case only the raw state_dict was saved
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    return model


def predict_image(model, image_path: Path, device: torch.device, transform) -> dict:
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu()

    top_idx = probs.argmax().item()
    top_conf = probs[top_idx].item()

    return {
        "file": image_path.name,
        "prediction": config.CLASS_NAMES[top_idx],
        "confidence": round(top_conf * 100, 2),
        "probs": {
            cls: round(probs[i].item() * 100, 2)
            for i, cls in enumerate(config.CLASS_NAMES)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Metal classifier inference")
    parser.add_argument("--model", required=True, help="Model name, e.g. resnet18")
    parser.add_argument("--image", default=None, help="Path to a single image")
    parser.add_argument("--folder", default=None, help="Path to a folder of images")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else config.OUTPUT_DIR / args.model / "model" / config.MODEL_NAME
    )

    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}\nRun train.py first.")

    model = load_model(args.model, ckpt, device)
    transform = test_transform

    if args.image:
        result = predict_image(model, Path(args.image), device, transform)
        print(f"\nFile       : {result['file']}")
        print(f"Prediction : {result['prediction']}  ({result['confidence']:.2f}%)")
        print("All probs  :")
        for cls, p in sorted(result["probs"].items(), key=lambda x: -x[1]):
            bar = "█" * int(p / 5)
            print(f"  {cls:12s} {p:6.2f}%  {bar}")

    elif args.folder:
        folder = Path(args.folder)
        images = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        print(f"\nRunning inference on {len(images)} images in {folder}\n")
        for img_path in images:
            r = predict_image(model, img_path, device, transform)
            print(
                f"  {r['file']:40s}  → {r['prediction']:12s}  ({r['confidence']:.1f}%)"
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
