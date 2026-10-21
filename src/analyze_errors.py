
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "best_mobilenet.keras"
CLASSES_PATH = ROOT / "models" / "mobilenet_class_names.json"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"
OUTPUT_DIR = ROOT / "reports" / "mobilenet"

IMAGE_SIZE = (224, 224)


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(
        f"Could not find any of {candidates}. "
        f"CSV columns are: {list(df.columns)}"
    )


def main():
    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)
    df = pd.read_csv(TEST_CSV)

    path_col = find_column(
        df, ["path", "image_path", "filepath", "file_path", "image"]
    )
    label_col = find_column(
        df, ["label", "class", "species", "class_name"]
    )

    records = []

    for row_number, row in df.iterrows():
        image_path = Path(str(row[path_col]))

        if not image_path.is_absolute():
            image_path = ROOT / image_path

        actual = str(row[label_col])

        try:
            
            image_bytes = tf.io.read_file(str(image_path))
            image = tf.io.decode_image(
                image_bytes,
                channels=3,
                expand_animations=False
            )
            image.set_shape([None, None, 3])

            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(
                image
            )

            array = image.numpy()
            array = np.expand_dims(array, axis=0)


            scores = model.predict(array, verbose=0)[0]
            order = np.argsort(scores)[::-1]

            predicted_index = int(order[0])
            predicted = class_names[predicted_index]

            record = {
                "image_path": str(image_path),
                "actual_label": actual,
                "predicted_label": predicted,
                "confidence": float(scores[predicted_index]),
                "correct": actual == predicted,
            }

            for rank, index in enumerate(order[:3], start=1):
                record[f"top{rank}_label"] = class_names[int(index)]
                record[f"top{rank}_confidence"] = float(scores[index])

            records.append(record)

        except Exception as exc:
            print(f"Skipped {image_path}: {exc}")

        if (row_number + 1) % 25 == 0:
            print(f"Processed {row_number + 1}/{len(df)} rows")

    results = pd.DataFrame(records)

    if results.empty:
        raise RuntimeError(
            "No images were processed. Check the CSV paths and model files."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "test_error_analysis.csv", index=False
    )

    accuracy = results["correct"].mean()

    print("\n--- Test Error Analysis ---")
    print(f"Images processed: {len(results)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Incorrect predictions: {(~results['correct']).sum()}")

    errors = results.loc[~results["correct"]]

    if errors.empty:
        print("No incorrect predictions in this run.")
    else:
        print("\nMost common actual -> predicted confusions:")
        confusion_counts = (
            errors.groupby(
                ["actual_label", "predicted_label"]
            )
            .size()
            .sort_values(ascending=False)
            .head(15)
        )
        print(confusion_counts.to_string())

        print("\nIncorrect predictions with lowest confidence:")
        print(
            errors[
                ["actual_label", "predicted_label", "confidence"]
            ]
            .sort_values("confidence")
            .head(10)
            .to_string(index=False)
        )

    print(
        f"\nDetailed results saved to: "
        f"{OUTPUT_DIR / 'test_error_analysis.csv'}"
    )


if __name__ == "__main__":
    main()
