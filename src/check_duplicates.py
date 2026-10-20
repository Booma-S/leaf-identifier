
from pathlib import Path
from collections import defaultdict
import hashlib

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"

CSV_FILES = {
    "train": SPLIT_DIR / "train.csv",
    "val": SPLIT_DIR / "val.csv",
    "test": SPLIT_DIR / "test.csv",
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


hash_locations = defaultdict(list)

for split_name, csv_path in CSV_FILES.items():
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        image_path = ROOT / row["image_path"]
        image_hash = sha256_file(image_path)

        hash_locations[image_hash].append({
            "split": split_name,
            "path": row["image_path"],
            "label": row["label"],
        })

cross_split_duplicates = []

for locations in hash_locations.values():
    split_names = {item["split"] for item in locations}

    if len(split_names) > 1:
        cross_split_duplicates.append(locations)

print(f"Unique image contents: {len(hash_locations)}")
print(f"Cross-split duplicate groups: {len(cross_split_duplicates)}")

if cross_split_duplicates:
    print("\nDuplicates found across splits:")

    for group in cross_split_duplicates:
        print("-" * 60)
        for item in group:
            print(
                f"Split: {item['split']} | "
                f"Label: {item['label']} | "
                f"Path: {item['path']}"
            )

    print(
        "\nReview these groups before training. "
        "Do not automatically delete images."
    )
else:
    print("No exact duplicate files found across splits.")
