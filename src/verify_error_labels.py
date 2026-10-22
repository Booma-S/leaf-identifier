
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

ERRORS_CSV = ROOT / "reports" / "mobilenet" / "test_error_analysis.csv"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing expected columns. Found: {list(df.columns)}")


def resolve_path(value):
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def main():
    errors = pd.read_csv(ERRORS_CSV)
    test = pd.read_csv(TEST_CSV)

    test_path_col = find_column(
        test, ["path", "image_path", "filepath", "file_path", "image"]
    )
    test_label_col = find_column(
        test, ["label", "class", "species", "class_name"]
    )

    error_path_col = find_column(
        errors, ["image_path", "path", "image"]
    )

    error_label_col = find_column(
        errors, ["actual_label", "label", "class"]
    )

    incorrect = errors.loc[~errors["correct"].astype(bool)].copy()

    # Match paths using normalized absolute paths.
    test["_resolved_path"] = test[test_path_col].apply(
        lambda p: str(resolve_path(p).resolve()).casefold()
    )
    incorrect["_resolved_path"] = incorrect[error_path_col].apply(
        lambda p: str(resolve_path(p).resolve()).casefold()
    )

    print("\n--- Verify Misclassified Test Images ---\n")

    for _, row in incorrect.iterrows():
        matches = test.loc[
            test["_resolved_path"] == row["_resolved_path"]
        ]

        print(f"Image: {row[error_path_col]}")
        print(f"Actual label in error report: {row[error_label_col]}")

        if len(matches) == 1:
            source_label = str(matches.iloc[0][test_label_col])
            print(f"Label in test.csv: {source_label}")
            print(f"Labels match: {source_label == str(row[error_label_col])}")
        else:
            print(f"Test CSV matches found: {len(matches)}")

        image_path = resolve_path(row[error_path_col])
        print(f"Image file exists: {image_path.is_file()}")

        if image_path.is_file():
            try:
                with Image.open(image_path) as image:
                    image.verify()
                print("Image file readable: Yes")
            except Exception as exc:
                print(f"Image file readable: No ({exc})")

        print("-" * 60)


if __name__ == "__main__":
    main()
