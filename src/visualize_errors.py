
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "reports" / "mobilenet" / "test_error_analysis.csv"
OUTPUT_PATH = ROOT / "reports" / "mobilenet" / "misclassified_contact_sheet.png"


def main():
    df = pd.read_csv(CSV_PATH)

    errors = df.loc[~df["correct"].astype(bool)].copy()

    if errors.empty:
        print("No misclassified images found.")
        return

    columns = 3
    rows = (len(errors) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows, columns, figsize=(15, 5 * rows)
    )
    axes = list(axes.flat)

    for ax, (_, row) in zip(axes, errors.iterrows()):
        image_path = Path(row["image_path"])

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        image = ImageOps.exif_transpose(
            Image.open(image_path)
        ).convert("RGB")

        ax.imshow(image)
        ax.set_title(
            f"Actual: {row['actual_label']}\n"
            f"Predicted: {row['predicted_label']}\n"
            f"Confidence: {row['confidence']:.1%}",
            fontsize=9
        )
        ax.axis("off")

    for ax in axes[len(errors):]:
        ax.axis("off")

    fig.suptitle(
        "MobileNetV2 — Misclassified Test Images",
        fontsize=15
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Misclassified images: {len(errors)}")
    print(f"Contact sheet saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
