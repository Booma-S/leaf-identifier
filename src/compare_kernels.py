
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "reports" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# One representative image from each of five species.
SAMPLES = [
    ("Alpinia Galanga (Rasna)", "AG-S-001.jpg"),
    ("Amaranthus Viridis (Arive-Dantu)", "AV-S-001.jpg"),
    ("Ficus Religiosa (Peepal Tree)", "FR-S-001.jpg"),
    ("Mentha (Mint)", "M-S-001.jpg"),
    ("Plectranthus Amboinicus (Mexican Mint)", "PA-S-001.jpg"),
]


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    if count <= 1:
        return np.zeros_like(mask)

    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == label, 255, 0).astype(np.uint8)


def fill_holes(mask):
    result = mask.copy()
    h, w = result.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(result, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(result)
    return cv2.bitwise_or(mask, holes)


def get_initial_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]

    _, sat_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, dark_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return cv2.bitwise_or(sat_mask, dark_mask)


def segment_with_kernel(initial, kernel_size):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    opened = cv2.morphologyEx(
        initial, cv2.MORPH_OPEN, kernel
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, kernel
    )

    component = largest_component(closed)
    return fill_holes(component)


def main():
    rows = []

    fig, axes = plt.subplots(
        len(SAMPLES), 3, figsize=(13, 4 * len(SAMPLES))
    )

    for i, (species, filename) in enumerate(SAMPLES):
        image_path = DATA_DIR / species / filename

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read: {image_path}")
            continue

        initial = get_initial_mask(image)
        mask3 = segment_with_kernel(initial, 3)
        mask5 = segment_with_kernel(initial, 5)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{species}\nOriginal")

        axes[i, 1].imshow(mask3, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title("3 × 3 kernel")

        axes[i, 2].imshow(mask5, cmap="gray", vmin=0, vmax=255)
        axes[i, 2].set_title("5 × 5 kernel")

        for ax in axes[i]:
            ax.axis("off")

        for size, mask in [(3, mask3), (5, mask5)]:
            rows.append({
                "species": species,
                "image": filename,
                "kernel_size": size,
                "foreground_fraction": np.mean(mask > 0),
            })

        print(
            f"{species}: "
            f"3x3={np.mean(mask3 > 0):.2%}, "
            f"5x5={np.mean(mask5 > 0):.2%}"
        )

    fig.suptitle("Morphological Kernel Comparison", fontsize=16)
    fig.tight_layout()

    output = OUTPUT_DIR / "kernel_comparison.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)

    import csv

    csv_path = OUTPUT_DIR / "kernel_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFigure saved to: {output}")
    print(f"Measurements saved to: {csv_path}")


if __name__ == "__main__":
    main()
