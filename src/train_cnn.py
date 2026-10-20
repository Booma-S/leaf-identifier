
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 15
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# --------------------------------------------------
# Load CSV files
# --------------------------------------------------
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "val.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

classes = sorted(train_df["label"].unique())
num_classes = len(classes)
class_to_index = {name: i for i, name in enumerate(classes)}

assert set(val_df["label"]) == set(classes)
assert set(test_df["label"]) == set(classes)

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Testing images: {len(test_df)}")
print(f"Number of classes: {num_classes}")

# --------------------------------------------------
# Image loading
# --------------------------------------------------
def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def make_dataset(df, training=False):
    paths = [
        str(ROOT / p)
        for p in df["image_path"]
    ]

    labels = np.array(
        [class_to_index[name] for name in df["label"]],
        dtype=np.int32,
    )

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# --------------------------------------------------
# Build a baseline CNN
# --------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3)),

    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# Train and save the best validation model
# --------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)

best_model_path = MODEL_DIR / "best_leaf_cnn.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save class names in the exact order used by the model.
with (MODEL_DIR / "class_names.json").open("w", encoding="utf-8") as file:
    json.dump(classes, file, indent=2, ensure_ascii=False)

# Save training history for later plots and analysis.
history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", range(1, len(history_df) + 1))
history_df.to_csv(MODEL_DIR / "training_history.csv", index=False)

print("\nTraining completed.")
print(f"Best model: {best_model_path}")
print(f"Class names: {MODEL_DIR / 'class_names.json'}")
print(f"Training history: {MODEL_DIR / 'training_history.csv'}")

