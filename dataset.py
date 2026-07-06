from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from config import CLASS_NAMES


class ThermalDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_to_idx = {
            class_name: idx for idx, class_name in enumerate(CLASS_NAMES)
        }

        self.samples = []

        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                ]:
                    continue

                self.samples.append((image_path, self.class_to_idx[class_name]))

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label