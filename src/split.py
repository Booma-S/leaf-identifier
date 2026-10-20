
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Project root: one level above this script's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main():
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {RAW_DIR}")

    rows = []

    # Each immediate subfolder represents one species.
    for species_dir in sorted(RAW_DIR.iterdir()):
        if not species_dir.is_dir():
            continue

        images = sorted(
            p for p in species_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in images:
            rows.append({
                "image_path": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "label": species_dir.name,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No images found in {RAW_DIR}")

    # Check that each species has enough images for all three splits.
    counts = df["label"].value_counts()
    too_small = counts[counts < 3]

    if not too_small.empty:
        raise ValueError(
            "Every species needs at least 3 images for stratified splitting:\n"
            + too_small.to_string()
        )

    # First reserve 15% for testing.
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )

    # The remaining 85% is split so validation is 15% of the full dataset.
    train, val = train_test_split(
        train_val,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=RANDOM_SEED,
        stratify=train_val["label"],
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    val.to_csv(SPLIT_DIR / "val.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)

    # Verify no image path appears in more than one split.
    train_paths = set(train["image_path"])
    val_paths = set(val["image_path"])
    test_paths = set(test["image_path"])

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    # Confirm every split contains the same set of species.
    expected_labels = set(df["label"])
    assert set(train["label"]) == expected_labels
    assert set(val["label"]) == expected_labels
    assert set(test["label"]) == expected_labels

    print("\nDataset split completed successfully!")
    print(f"Total images: {len(df)}")
    print(f"Species: {df['label'].nunique()}")
    print(f"Random seed: {RANDOM_SEED}")

    print("\nSplit summary:")
    for name, split in [
        ("Training", train),
        ("Validation", val),
        ("Testing", test),
    ]:
        print(
            f"{name:12} {len(split):5} images "
            f"({len(split) / len(df) * 100:.2f}%)"
        )

    print(f"\nCSV files saved in: {SPLIT_DIR}")

    print("\nImages per species in each split (first 10 species):")
    summary = pd.DataFrame({
        "train": train["label"].value_counts(),
        "validation": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0).astype(int)

    print(summary.sort_index().head(10).to_string())


if __name__ == "__main__":
    main()
