
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent

ERRORS_CSV = ROOT / "reports" / "mobilenet" / "test_error_analysis.csv"
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
OUTPUT_PATH = ROOT / "reports" / "mobilenet" / "error_training_comparison.png"

PAIRS = [
    ("Murraya Koenigii (Curry)", "Santalum Album (Sandalwood)"),
    ("Trigonella Foenum-graecum (Fenugreek)", "Amaranthus Viridis (Arive-Dantu)"),
    ("Trigonella Foenum-graecum (Fenugreek)", "Moringa Oleifera (Drumstick)"),
    ("Ocimum Tenuiflorum (Tulsi)", "Jasminum (Jasmine)"),
    ("Syzygium Cumini (Jamun)", "Tabernaemontana Divaricata (Crape Jasmine)"),
]


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing expected columns. Available: {list(df.columns)}")


def resolve_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_image(path):
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB").copy()


def main():
    errors = pd.read_csv(ERRORS_CSV)
    train = pd.read_csv(TRAIN_CSV)

    error_path_col = find_column(errors, ["image_path", "path", "image"])
    actual_col = find_column(errors, ["actual_label", "label", "class"])
    predicted_col = find_column(errors, ["predicted_label", "prediction"])

    train_path_col = find_column(
        train, ["path", "image_path", "filepath", "file_path", "image"]
    )
    train_label_col = find_column(
        train, ["label", "class", "species", "class_name"]
    )

    errors = errors.loc[~errors["correct"].astype(bool)]

    if errors.empty:
        print("No misclassified images found.")
        return

    panels = []

    for _, error in errors.iterrows():
        actual = str(error[actual_col])
        predicted = str(error[predicted_col])

        panels.append((
            resolve_path(error[error_path_col]),
            f"TEST ERROR\nActual: {actual}\nPredicted: {predicted}",
        ))

        for label in [actual, predicted]:
            candidates = train.loc[train[train_label_col].astype(str) == label]

            if candidates.empty:
                print(f"No training rows found for class: {label}")
                continue

            # Pick up to two examples spread through the class rows.
            positions = sorted(set([
                0,
                len(candidates) // 2,
                len(candidates) - 1,
            ]))
            for pos in positions[:2]:
                row = candidates.iloc[positions[pos] if False else pos]
                panels.append((
                    resolve_path(row[train_path_col]),
                    f"TRAINING EXAMPLE\nClass: {label}",
                ))

    columns = 4
    rows = (len(panels) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows))
    axes = list(axes.flat)

    for ax, (image_path, title) in zip(axes, panels):
        try:
            ax.imshow(load_image(image_path))
            ax.set_title(title, fontsize=8, wrap=True)
        except Exception as exc:
            ax.text(0.5, 0.5, f"Could not load image:\n{exc}",
                    ha="center", va="center", wrap=True)
        ax.axis("off")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.suptitle("Misclassified Test Images and Training Examples", fontsize=15)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison panels: {len(panels)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
