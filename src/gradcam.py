
from pathlib import Path
import json
import argparse

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_mobilenet.keras"
CLASS_NAMES_PATH = ROOT / "models" / "mobilenet_class_names.json"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"
OUTPUT_DIR = ROOT / "reports" / "gradcam"

IMAGE_SIZE = (224, 224)


def load_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)




def load_model_and_grad_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    base_model = model.get_layer("mobilenetv2_1.00_224")

    # Use a higher-resolution internal feature map (14 x 14).
    last_conv_layer = base_model.get_layer("block_13_expand_relu")
    print(
        f"Using Grad-CAM layer: {last_conv_layer.name} "
        f"{last_conv_layer.output.shape}"
    )

    # Keep the selected feature map connected to the backbone output.
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output,
        ],
    )

    pooling_layer = model.get_layer("global_average_pooling2d")
    dropout_layer = model.get_layer("dropout")
    classifier_layer = model.get_layer("dense")

    def grad_model(image_tensor, training=False):
        feature_maps, backbone_output = feature_model(
            image_tensor, training=training
        )

        x = pooling_layer(backbone_output)
        x = dropout_layer(x, training=False)
        predictions = classifier_layer(x)

        return feature_maps, predictions

    return model, base_model, last_conv_layer.name, grad_model




def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)

    # Same preprocessing used during MobileNetV2 training.
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return original, tf.expand_dims(array, axis=0)


def make_gradcam_heatmap(image_tensor, grad_model):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        predicted_index = tf.argmax(predictions[0])
        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, conv_outputs)

    # Average gradients over the spatial dimensions.
    weights = tf.reduce_mean(gradients, axis=(1, 2))

    # Weight each feature map and combine them.
    conv_outputs = conv_outputs[0]
    weights = weights[0]

    heatmap = tf.reduce_sum(
        conv_outputs * weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    # Keep positive contributions.
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)

    heatmap = tf.where(
        max_value > 0,
        heatmap / (max_value + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy(), int(predicted_index.numpy()), predictions.numpy()[0]


def save_visualization(original, heatmap, class_name, confidence, output_path):
    # Resize the heatmap to the original image dimensions.
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        original.shape[:2],
    ).numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original image")
    axes[0].axis("off")

    axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")

    axes[2].imshow(original)
    axes[2].imshow(
        heatmap_resized,
        cmap="jet",
        alpha=0.30,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {class_name} ({confidence:.2%})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for MobileNetV2."
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image to explain. Relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=4,
        help="Number of test images to use when --image is not supplied.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()
    model, base_model, conv_name, grad_model = load_model_and_grad_model()

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        image_paths = [image_path]
    else:
        test_df = pd.read_csv(TEST_CSV)
        image_paths = [
            ROOT / path
            for path in test_df["image_path"].head(args.samples)
        ]

    for index, image_path in enumerate(image_paths, start=1):
        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        original, image_tensor = preprocess_image(image_path)
        heatmap, predicted_index, probabilities = make_gradcam_heatmap(
            image_tensor, grad_model
        )

        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        output_path = OUTPUT_DIR / f"gradcam_{index:02d}.png"
        save_visualization(
            original,
            heatmap,
            predicted_class,
            confidence,
            output_path,
        )

        print(f"\nImage: {image_path}")
        print(f"Predicted species: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Saved: {output_path}")

    print(f"\nGrad-CAM outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
