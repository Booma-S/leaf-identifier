
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ERRORS_CSV = ROOT / "reports" / "mobilenet" / "test_error_analysis.csv"
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Expected one of {candidates}; found {list(df.columns)}")


def main():
    errors_df = pd.read_csv(ERRORS_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_label_col = find_column(
        train_df, ["label", "class", "species", "class_name"]
    )
    test_label_col = find_column(
        test_df, ["label", "class", "species", "class_name"]
    )

    errors = errors_df.loc[~errors_df["correct"].astype(bool)].copy()

    train_counts = train_df[train_label_col].value_counts()
    test_counts = test_df[test_label_col].value_counts()

    rows = []

    for species in sorted(set(errors["actual_label"])):
        class_errors = errors.loc[errors["actual_label"] == species]

        rows.append({
            "species": species,
            "training_images": int(train_counts.get(species, 0)),
            "test_images": int(test_counts.get(species, 0)),
            "incorrect_predictions": len(class_errors),
            "test_errors_pct": (
                len(class_errors) / test_counts.get(species, 1) * 100
            ),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("incorrect_predictions", ascending=False)

    print("\n--- Classes with Misclassified Test Images ---")
    print(result.to_string(index=False, formatters={
        "test_errors_pct": "{:.2f}%".format
    }))

    output = ROOT / "reports" / "mobilenet" / "misclassified_class_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    print(f"\nSaved summary to: {output}")


if __name__ == "__main__":
    main()
