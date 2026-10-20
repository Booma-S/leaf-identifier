
from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MASK_DIR = ROOT / "data" / "processed" / "masks"
REPORT_DIR = ROOT / "reports" / "segmentation"
PREVIEW_DIR = ROOT / "reports" / "segmentation_preview"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def fill_holes(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    flood = mask.copy()
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )

    if count <= 1:
        return np.zeros_like(mask)

    label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == label, 255, 0).astype(np.uint8)


def segment_leaf(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    _, sat_mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    _, dark_mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    combined = cv2.bitwise_or(sat_mask, dark_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    component = largest_component(closed)
    mask = fill_holes(component)

    stages = {
        "Gray": gray,
        "Saturation": saturation,
        "Combined Otsu": combined,
        "Cleaned mask": mask,
    }
    return mask, stages


def make_preview(image_path, image, mask, stages, output_path):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    foreground = mask > 0
    overlay[foreground] = (
        0.65 * overlay[foreground]
        + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)

    panels = [
        ("Original", rgb),
        ("Grayscale", stages["Gray"]),
        ("Saturation", stages["Saturation"]),
        ("Combined Otsu", stages["Combined Otsu"]),
        ("Cleaned mask", mask),
        ("Overlay", overlay),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, panel) in zip(axes.flat, panels):
        ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(image_path.name)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def process_image(image_path, make_figure=False):
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "unreadable", None

    mask, stages = segment_leaf(image)
    fraction = float(np.count_nonzero(mask) / mask.size)

    warnings = []
    if fraction < 0.01:
        warnings.append("foreground_below_1_percent")
    if fraction > 0.90:
        warnings.append("foreground_above_90_percent")
    if not np.any(mask):
        warnings.append("empty_mask")

    relative_path = image_path.relative_to(DATA_DIR)
    mask_path = (MASK_DIR / relative_path).with_suffix(".png")
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(mask_path), mask):
        return fraction, "mask_write_failed", mask_path

    if make_figure:
        preview_path = (
            PREVIEW_DIR / relative_path.parent
            / f"{image_path.stem}_stages.png"
        )
        make_preview(image_path, image, mask, stages, preview_path)

    status = ";".join(warnings) if warnings else "ok"
    return fraction, status, mask_path


def run_single(image_path):
    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    fraction, status, mask_path = process_image(
        image_path, make_figure=True
    )
    if fraction is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    print(f"Image: {image_path}")
    print(f"Foreground: {fraction:.2%}")
    print(f"Status: {status}")
    print(f"Mask: {mask_path}")


def run_batch():
    images = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(f"No images found in {DATA_DIR}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORT_DIR / "segmentation_log.csv"
    flagged_path = REPORT_DIR / "flagged_masks.csv"

    flagged = []
    successful = 0

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "image", "foreground_fraction", "status", "mask_path"
            ],
        )
        writer.writeheader()

        for index, image_path in enumerate(images, start=1):
            fraction, status, mask_path = process_image(image_path)

            if fraction is None:
                successful_status = status
                fraction_text = ""
                mask_text = ""
            else:
                successful += 1
                successful_status = status
                fraction_text = f"{fraction:.6f}"
                mask_text = str(mask_path.relative_to(ROOT))

            row = {
                "image": str(image_path.relative_to(ROOT)),
                "foreground_fraction": fraction_text,
                "status": successful_status,
                "mask_path": mask_text,
            }
            writer.writerow(row)

            if successful_status != "ok":
                flagged.append(row)

            if index % 100 == 0 or index == len(images):
                print(f"Processed {index}/{len(images)}")

    with flagged_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image", "foreground_fraction", "status", "mask_path"
            ],
        )
        writer.writeheader()
        writer.writerows(flagged)

    print("\nBatch segmentation finished.")
    print(f"Images found: {len(images)}")
    print(f"Images processed successfully: {successful}")
    print(f"Flagged for review: {len(flagged)}")
    print(f"Flag rate: {len(flagged) / len(images):.2%}")
    print(f"Log: {log_path}")
    print(f"Flagged list: {flagged_path}")
    print(f"Masks: {MASK_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", type=Path,
        help="Process one image instead of the full dataset."
    )
    args = parser.parse_args()

    if args.image:
        run_single(args.image)
    else:
        run_batch()
