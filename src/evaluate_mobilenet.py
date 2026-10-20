
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

TEST_CSV = ROOT / "data" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "best_mobilenet.keras"
CLASS_NAMES_PATH = ROOT / "models" / "mobilenet_class_names.json"
OUTPUT_DIR = ROOT / "reports" / "mobilenet"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load test data
df = pd.read_csv(TEST_CSV)

print(f"Test images: {len(df)}")

# Check the CSV column names if this fails.
print("CSV columns:", df.columns.tolist())

# This assumes the split CSV contains 'path' and 'label' columns.
# If your CSV uses different names, adjust these two lines.
image_paths = df["image_path"].astype(str).tolist()
true_labels = df["label"].astype(str).tolist()

# Load the saved class order
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

# Handle either a list of class names or a dictionary mapping
# class indices to class names.
if isinstance(class_names, dict):
    class_names = [
        class_names[str(i)] for i in range(len(class_names))
    ]

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

unknown_labels = sorted(set(true_labels) - set(class_to_index))
if unknown_labels:
    raise ValueError(f"Labels missing from class names: {unknown_labels}")

y_true = np.array([class_to_index[label] for label in true_labels])

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

paths_tensor = tf.constant(image_paths)
labels_tensor = tf.constant(y_true, dtype=tf.int32)

test_ds = tf.data.Dataset.from_tensor_slices(
    (paths_tensor, labels_tensor)
)
test_ds = test_ds.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Load model and predict
model = tf.keras.models.load_model(MODEL_PATH)

test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\n--- MobileNetV2 Test Results ---")
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy (%): {test_accuracy * 100:.2f}%")

# Classification report
report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

# Confusion matrix
cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(len(class_names))
)
pd.DataFrame(
    cm, index=class_names, columns=class_names
).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(16, 14))
image = ax.imshow(cm, interpolation="nearest")
ax.set_title("MobileNetV2 — Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))
ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_yticklabels(class_names, fontsize=7)
fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

# Save a compact metrics summary
summary = pd.DataFrame([{
    "model": "MobileNetV2",
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "macro_precision": report["macro avg"]["precision"],
    "macro_recall": report["macro avg"]["recall"],
    "macro_f1": report["macro avg"]["f1-score"],
    "weighted_f1": report["weighted avg"]["f1-score"],
}])
summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

print("\nSaved evaluation files to:", OUTPUT_DIR)
print(summary.to_string(index=False))
