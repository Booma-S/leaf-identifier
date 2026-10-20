
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"

files = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}

splits = {}

for name, csv_path in files.items():
    df = pd.read_csv(csv_path)

    assert {"image_path", "label"}.issubset(df.columns), (
        f"{csv_path.name} must contain image_path and label columns"
    )
    assert not df.empty, f"{csv_path.name} is empty"
    assert not df["image_path"].duplicated().any(), (
        f"Duplicate paths found inside {name}"
    )

    # Verify that every referenced image exists.
    missing = [
        p for p in df["image_path"]
        if not (ROOT / p).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{name}: {len(missing)} missing images. Examples: {missing[:5]}"
        )

    splits[name] = df
    print(f"{name}: {len(df)} valid image paths")

# Ensure the same path does not occur across different splits.
path_sets = {
    name: set(df["image_path"])
    for name, df in splits.items()
}

assert path_sets["train"].isdisjoint(path_sets["val"])
assert path_sets["train"].isdisjoint(path_sets["test"])
assert path_sets["val"].isdisjoint(path_sets["test"])

# Ensure each split contains all species.
label_sets = {
    name: set(df["label"])
    for name, df in splits.items()
}

assert label_sets["train"] == label_sets["val"] == label_sets["test"], (
    "Species labels differ between splits"
)

print("\nAll checks passed!")
print("No overlapping image paths.")
print("All image paths exist.")
print(f"Species represented in every split: {len(label_sets['train'])}")
