
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# Load splits and class order
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "val.csv")

classes = sorted(train_df["label"].unique())
class_to_index = {name: i for i, name in enumerate(classes)}
num_classes = len(classes)

assert set(val_df["label"]) == set(classes)

print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Classes: {num_classes}")


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)

    # MobileNetV2 expects pixel values scaled to [-1, 1].
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


def make_dataset(df, training=False):
    paths = [str(ROOT / p) for p in df["image_path"]]
    labels = np.array(
        [class_to_index[name] for name in df["label"]],
        dtype=np.int32,
    )

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(df), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds.prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)

# Load pretrained MobileNetV2 without its original classifier.
base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3),
)

# Freeze pretrained feature extractor for initial training.
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    num_classes, activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

MODEL_DIR.mkdir(parents=True, exist_ok=True)
best_model_path = MODEL_DIR / "best_mobilenet.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(best_model_path),
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

# Save the class order used by this model.
with (MODEL_DIR / "mobilenet_class_names.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(classes, file, indent=2, ensure_ascii=False)

history_df = pd.DataFrame(history.history)
history_df.insert(0, "epoch", range(1, len(history_df) + 1))
history_df.to_csv(
    MODEL_DIR / "mobilenet_training_history.csv", index=False
)

print("\nMobileNetV2 training complete.")
print(f"Best model: {best_model_path}")
print(f"Class names: {MODEL_DIR / 'mobilenet_class_names.json'}")
print(f"Training history: {MODEL_DIR / 'mobilenet_training_history.csv'}")
