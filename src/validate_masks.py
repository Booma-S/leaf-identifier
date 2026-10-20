
from pathlib import Path
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MASK_DIR = ROOT / "data" / "processed" / "masks"
OUTPUT = ROOT / "reports" / "segmentation" / "visual_validation_sample.png"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
rng = random.Random(42)  # Reproducible sample

samples = []

for folder in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        continue

    image_path = rng.choice(images)
    relative = image_path.relative_to(DATA_DIR)
    mask_path = (MASK_DIR / relative).with_suffix(".png")

    if not mask_path.exists():
        print(f"Missing mask: {relative}")
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Could not read: {relative}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize for the contact sheet
    image = cv2.resize(image, (240, 180))
    mask = cv2.resize(mask, (240, 180), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.65 * overlay[foreground]
        + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)

    samples.append((folder.name, image, mask, overlay, relative))

cols = 5
rows = int(np.ceil(len(samples) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.4))
axes = np.array(axes).reshape(-1)

for ax, (species, image, mask, overlay, relative) in zip(axes, samples):
    ax.imshow(overlay)
    ax.set_title(species, fontsize=8)
    ax.axis("off")

for ax in axes[len(samples):]:
    ax.axis("off")

fig.suptitle(
    f"Segmentation Visual Validation — {len(samples)} Species",
    fontsize=15
)
fig.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Species inspected: {len(samples)}")
print(f"Contact sheet saved to: {OUTPUT}")
print("Red overlay indicates the predicted foreground mask.")
