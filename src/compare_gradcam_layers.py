
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_mobilenet.keras"
CLASS_NAMES_PATH = ROOT / "models" / "mobilenet_class_names.json"
TEST_CSV = ROOT / "data" / "splits" / "test.csv"

OUTPUT_DIR = ROOT / "reports" / "gradcam" / "layer_comparison"

IMAGE_SIZE = (224, 224)
LAYERS_TO_COMPARE = [
    "Conv_1",
    "block_13_expand_relu",
    "block_13_depthwise_relu",
]


def load_image(path):
    image = Image.open(path).convert("RGB")
    original = np.asarray(image)
    resized = image.resize(IMAGE_SIZE)
    array = np.asarray(resized, dtype=np.float32)

    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return original, tf.expand_dims(array, axis=0)



def build_grad_model(model, layer_name):
    base = model.get_layer("mobilenetv2_1.00_224")
    target_layer = base.get_layer(layer_name)

    # Build the model before the selected feature layer.
    prefix_model = tf.keras.Model(
        inputs=base.input,
        outputs=target_layer.output,
    )

    # Continue from that feature layer to the backbone output.
    suffix_model = tf.keras.Model(
        inputs=target_layer.output,
        outputs=base.output,
    )

    pooling = model.get_layer("global_average_pooling2d")
    dropout = model.get_layer("dropout")
    classifier = model.get_layer("dense")

    return prefix_model, suffix_model, pooling, dropout, classifier


def generate_heatmap(image_tensor, grad_parts):
    prefix_model, suffix_model, pooling, dropout, classifier = grad_parts

    # Keep the entire path from the selected feature map to the score
    # inside GradientTape so gradients can be calculated.
    with tf.GradientTape() as tape:
        feature_maps = prefix_model(image_tensor, training=False)
        tape.watch(feature_maps)

        backbone_output = suffix_model(feature_maps, training=False)
        pooled = pooling(backbone_output)
        dropped = dropout(pooled, training=False)
        predictions = classifier(dropped)

        class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Gradients are disconnected for this layer. "
            "Please share the full traceback."
        )

    # Average gradients over the spatial dimensions.
    weights = tf.reduce_mean(gradients, axis=(1, 2))[0]
    maps = feature_maps[0]

    heatmap = tf.reduce_sum(
        maps * weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    heatmap = tf.where(
        maximum > 0,
        heatmap / (maximum + tf.keras.backend.epsilon()),
        tf.zeros_like(heatmap),
    )

    return heatmap.numpy(), int(class_index.numpy()), predictions.numpy()[0]



def overlay_image(original, heatmap):
    resized_heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        original.shape[:2],
    ).numpy().squeeze()

    return resized_heatmap


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    test_df = pd.read_csv(TEST_CSV).head(4)

    records = []

    for image_number, row in enumerate(
        test_df.itertuples(index=False), start=1
    ):
        image_path = ROOT / row.image_path

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        original, image_tensor = load_image(image_path)

        fig, axes = plt.subplots(
            len(LAYERS_TO_COMPARE), 3, figsize=(12, 10)
        )

        fig.suptitle(
            f"Grad-CAM layer comparison — image {image_number}\n"
            f"True label: {row.label}",
            fontsize=12,
        )

        for layer_number, layer_name in enumerate(LAYERS_TO_COMPARE):
            grad_parts = build_grad_model(model, layer_name)
            heatmap, pred_index, probabilities = generate_heatmap(
                image_tensor, grad_parts
            )

            predicted_name = class_names[pred_index]
            confidence = float(probabilities[pred_index])
            resized_heatmap = overlay_image(original, heatmap)

            axes[layer_number, 0].imshow(original)
            axes[layer_number, 0].set_title("Original")
            axes[layer_number, 0].axis("off")

            axes[layer_number, 1].imshow(
                resized_heatmap, cmap="jet", vmin=0, vmax=1
            )
            axes[layer_number, 1].set_title(
                f"{layer_name}\nHeatmap"
            )
            axes[layer_number, 1].axis("off")

            axes[layer_number, 2].imshow(original)
            axes[layer_number, 2].imshow(
                resized_heatmap,
                cmap="jet",
                alpha=0.30,
                vmin=0,
                vmax=1,
            )
            axes[layer_number, 2].set_title(
                f"{predicted_name}\nConfidence: {confidence:.2%}"
            )
            axes[layer_number, 2].axis("off")

            records.append({
                "image_path": str(row.image_path),
                "true_label": row.label,
                "layer": layer_name,
                "predicted_label": predicted_name,
                "confidence": confidence,
            })

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = OUTPUT_DIR / f"comparison_{image_number:02d}.png"
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")

    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "layer_comparison.csv", index=False
    )

    print(f"\nComparison complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
