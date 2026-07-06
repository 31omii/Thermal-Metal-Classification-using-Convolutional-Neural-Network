import shutil
from config import DATA_DIR, TRAIN_SPLIT, VAL_SPLIT

train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "val"
test_dir = DATA_DIR / "test"

for folder in [train_dir, val_dir, test_dir]:
    folder.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

classes = [
    cls for cls in DATA_DIR.iterdir()
    if cls.is_dir() and cls.name not in ["train", "val", "test"]
]

for cls in sorted(classes):

    images = []

    for ext in IMAGE_EXTS:
        images.extend(cls.glob(f"*{ext}"))
        images.extend(cls.glob(f"*{ext.upper()}"))

    images = sorted(images)

    total = len(images)

    train_end = int(total * TRAIN_SPLIT)
    val_end = train_end + int(total * VAL_SPLIT)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    (train_dir / cls.name).mkdir(parents=True, exist_ok=True)
    (val_dir / cls.name).mkdir(parents=True, exist_ok=True)
    (test_dir / cls.name).mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.copy2(img, train_dir / cls.name / img.name)

    for img in val_images:
        shutil.copy2(img, val_dir / cls.name / img.name)

    for img in test_images:
        shutil.copy2(img, test_dir / cls.name / img.name)

    print(
        f"{cls.name:<12}"
        f" Train: {len(train_images):4d}"
        f" Val: {len(val_images):4d}"
        f" Test: {len(test_images):4d}"
    )

print("\nSequential dataset split completed.")