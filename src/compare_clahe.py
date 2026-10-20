
from pathlib import Path
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "reports" / "segmentation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same five images as the kernel experiment.
SAMPLES = [
    ("Alpinia Galanga (Rasna)", "AG-S-001.jpg"),
    ("Amaranthus Viridis (Arive-Dantu)", "AV-S-001.jpg"),
    ("Ficus Religiosa (Peepal Tree)", "FR-S-001.jpg"),
    ("Mentha (Mint)", "M-S-001.jpg"),
    ("Plectranthus Amboinicus (Mexican Mint)", "PA-S-001.jpg"),
]

KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (5, 5)
)

# CLAHE settings; keep fixed for a fair initial comparison.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    if count <= 1:
        return np.zeros_like(mask)

    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == label, 255, 0).astype(np.uint8)


def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    flood = mask.copy()
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)

    return cv2.bitwise_or(mask, holes)


def cleanup(mask):
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, KERNEL
    )
    closed = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, KERNEL
    )

    return fill_holes(largest_component(closed))


def otsu_masks(gray, saturation):
    _, sat_mask = cv2.threshold(
        saturation, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    _, dark_mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return cv2.bitwise_or(sat_mask, dark_mask)


def segment_baseline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )[:, :, 1]

    return cleanup(otsu_masks(gray, saturation))


def segment_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )[:, :, 1]

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID
    )
    enhanced_gray = clahe.apply(gray)

    # Keep saturation thresholding unchanged.
    # Only the grayscale/darkness threshold uses CLAHE.
    initial_mask = otsu_masks(enhanced_gray, saturation)

    return cleanup(initial_mask), enhanced_gray


def make_overlay(image, mask):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    foreground = mask > 0

    red = np.array([255, 0, 0])
    overlay[foreground] = (
        0.65 * overlay[foreground] + 0.35 * red
    ).astype(np.uint8)

    return overlay


def main():
    rows = []
    figure_rows = []

    for species, filename in SAMPLES:
        image_path = DATA_DIR / species / filename
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        baseline = segment_baseline(image)
        clahe_mask, enhanced_gray = segment_clahe(image)

        base_fraction = float(np.mean(baseline > 0))
        clahe_fraction = float(np.mean(clahe_mask > 0))

        # Fraction of pixels whose binary labels differ.
        difference = float(
            np.mean((baseline > 0) != (clahe_mask > 0))
        )

        rows.append({
            "species": species,
            "image": filename,
            "baseline_foreground_fraction": base_fraction,
            "clahe_foreground_fraction": clahe_fraction,
            "mask_pixel_difference_fraction": difference,
        })

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        figure_rows.append((
            species, rgb, enhanced_gray,
            baseline, clahe_mask
        ))

        print(
            f"{species}: "
            f"baseline={base_fraction:.2%}, "
            f"CLAHE={clahe_fraction:.2%}, "
            f"different_pixels={difference:.2%}"
        )

    if not rows:
        raise RuntimeError(
            "No images processed. Check SAMPLES and data/raw/ paths."
        )

    fig, axes = plt.subplots(
        len(figure_rows), 4,
        figsize=(16, 4 * len(figure_rows))
    )

    for i, (species, rgb, enhanced, baseline, clahe_mask) in enumerate(
        figure_rows
    ):
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"{species}\nOriginal")

        axes[i, 1].imshow(enhanced, cmap="gray")
        axes[i, 1].set_title("CLAHE grayscale")

        axes[i, 2].imshow(baseline, cmap="gray", vmin=0, vmax=255)
        axes[i, 2].set_title("Baseline mask")

        axes[i, 3].imshow(clahe_mask, cmap="gray", vmin=0, vmax=255)
        axes[i, 3].set_title("CLAHE mask")

        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle("Baseline vs CLAHE Segmentation", fontsize=16)
    fig.tight_layout()

    figure_path = OUTPUT_DIR / "clahe_comparison.png"
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    csv_path = OUTPUT_DIR / "clahe_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nCLAHE experiment complete.")
    print(f"Images compared: {len(rows)}")
    print(f"Figure: {figure_path}")
    print(f"Measurements: {csv_path}")
    print(
        "Note: pixel difference measures disagreement, "
        "not segmentation accuracy."
    )


if __name__ == "__main__":
    main()
