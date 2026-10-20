
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "cnn_baseline"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16

# Load the saved model and class order.
model = tf.keras.models.load_model(
    MODEL_DIR / "best_leaf_cnn.keras"
)

with (MODEL_DIR / "class_names.json").open(
    encoding="utf-8"
) as file:
    classes = json.load(file)

test_df = pd.read_csv(SPLIT_DIR / "test.csv")
class_to_index = {name: i for i, name in enumerate(classes)}

if set(test_df["label"]) != set(classes):
    raise ValueError("Test split does not contain the expected classes.")

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

paths = [
    str(ROOT / p) for p in test_df["image_path"]
]
y_true = np.array(
    [class_to_index[label] for label in test_df["label"]],
    dtype=np.int32,
)

test_ds = tf.data.Dataset.from_tensor_slices((paths, y_true))
test_ds = test_ds.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Evaluate the test set once with the selected checkpoint.
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(classes)),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

REPORT_DIR.mkdir(parents=True, exist_ok=True)
report_df.to_csv(REPORT_DIR / "classification_report.csv")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        zero_division=0,
    )
)

# Save confusion matrix as CSV and an image.
cm = confusion_matrix(
    y_true, y_pred, labels=np.arange(len(classes))
)

pd.DataFrame(
    cm, index=classes, columns=classes
).to_csv(REPORT_DIR / "confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax.set(
    title="Baseline CNN — Test Confusion Matrix",
    xlabel="Predicted species",
    ylabel="True species",
    xticks=np.arange(len(classes)),
    yticks=np.arange(len(classes)),
)

ax.set_xticklabels(classes, rotation=90, fontsize=7)
ax.set_yticklabels(classes, fontsize=7)
fig.tight_layout()

fig.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)

print(f"\nReports saved to: {REPORT_DIR}")
