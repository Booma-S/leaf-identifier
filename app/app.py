
import json
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "models" / "best_mobilenet.keras"
CLASS_NAMES_PATH = ROOT_DIR / "models" / "mobilenet_class_names.json"
KNOWLEDGE_PATH = ROOT_DIR / "knowledge" / "plants.json"

IMAGE_SIZE = (224, 224)
TOP_K = 5


# --------------------------------------------------
# Load model and supporting data
# --------------------------------------------------

for path in (MODEL_PATH, CLASS_NAMES_PATH, KNOWLEDGE_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

model = tf.keras.models.load_model(MODEL_PATH)

with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
    class_names = json.load(file)

with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
    plant_data = json.load(file)

if not isinstance(class_names, list):
    raise ValueError("Expected mobilenet_class_names.json to contain a list.")

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        f"Model outputs {model.output_shape[-1]} classes, "
        f"but {len(class_names)} class names were loaded."
    )

print(f"Model loaded: {MODEL_PATH.name}")
print(f"Number of classes: {len(class_names)}")
print(f"Knowledge-base entries: {len(plant_data)}")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):
    """Prepare an uploaded PIL image for the trained MobileNetV2 model."""
    if image is None:
        raise ValueError("Please upload a leaf image first.")

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32)

    # Must match tensorflow.keras.applications.mobilenet_v2.preprocess_input
    # used when training the MobileNetV2 model.
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    return np.expand_dims(image_array, axis=0)


# --------------------------------------------------
# Prediction and plant information
# --------------------------------------------------

def identify_leaf(image):
    if image is None:
        return (
            "Please upload a leaf image.",
            {},
            "No plant information to display yet."
        )

    try:
        input_batch = preprocess_image(image)
        predictions = model.predict(input_batch, verbose=0)[0]

        if len(predictions) != len(class_names):
            raise ValueError("Prediction count does not match class names.")

        top_indices = np.argsort(predictions)[::-1][:TOP_K]

        # Gradio Label displays class names and confidence scores.
        top_predictions = {
            class_names[int(index)]: float(predictions[index])
            for index in top_indices
        }

        best_index = int(top_indices[0])
        predicted_species = class_names[best_index]
        confidence = float(predictions[best_index])

        details = plant_data.get(predicted_species, {})

        common_name = details.get("common_name") or predicted_species
        scientific_name = details.get("scientific_name") or "Not added yet"
        description = details.get("description") or (
            "Description is being researched."
        )

        medicinal_uses = details.get("medicinal_uses") or []
        traditional_uses = details.get("traditional_uses") or []
        sources = details.get("sources") or []

        lines = [
            f"## {common_name}",
            f"**Predicted dataset class:** {predicted_species}",
            f"**Scientific name:** {scientific_name}",
            f"**Model confidence:** {confidence:.1%}",
            "",
            description,
        ]

        lines.extend(["", "### Documented medicinal uses"])
        if medicinal_uses:
            lines.extend(f"- {use}" for use in medicinal_uses)
        else:
            lines.append("Research pending.")

        lines.extend(["", "### Traditional uses"])
        if traditional_uses:
            lines.extend(f"- {use}" for use in traditional_uses)
        else:
            lines.append("Research pending.")

        evidence_note = details.get("evidence_note", "")
        if evidence_note:
            lines.extend(["", f"**Evidence note:** {evidence_note}"])

        lines.extend(["", "### Sources"])
        if sources:
            for source in sources:
                if isinstance(source, dict):
                    title = source.get("title", "Reference")
                    url = source.get("url", "")
                    if url:
                        lines.append(f"- [{title}]({url})")
                    else:
                        lines.append(f"- {title}")
                else:
                    lines.append(f"- {source}")
        else:
            lines.append("References have not been added yet.")

        lines.extend([
            "",
            "> The prediction is an image-classification result, not a "
            "confirmed botanical identification. Model confidence is not "
            "a guarantee of correctness. Do not use this result alone "
            "to decide whether a plant is safe to consume or use medicinally."
        ])

        return predicted_species, top_predictions, "\n".join(lines)

    except (ValueError, OSError, UnidentifiedImageError) as error:
        return (
            f"Unable to process image: {error}",
            {},
            "Try uploading a valid JPG, PNG or other supported image."
        )
    except Exception as error:
        print(f"Prediction error: {error}")
        return (
            "Prediction failed. Check the terminal for details.",
            {},
            "Please try another image."
        )


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(title="Leaf Identifier") as demo:
    gr.Markdown(
        """
        # 🌿 Leaf Identifier
        Upload a leaf image to predict its species and explore
        the available plant information.

        **Note:** This research prototype predicts among 30 trained
        species. Results may be unreliable for unfamiliar plants,
        complex backgrounds or images unlike the training data.
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                type="pil",
                label="Upload a leaf image"
            )
            predict_button = gr.Button(
                "Identify Leaf",
                variant="primary"
            )

        with gr.Column():
            prediction_output = gr.Textbox(
                label="Top predicted species",
                interactive=False
            )
            confidence_output = gr.Label(
                label="Top 5 predictions",
                num_top_classes=TOP_K
            )

    plant_information = gr.Markdown(
        "Plant information will appear here after prediction."
    )

    predict_button.click(
        fn=identify_leaf,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output,
            plant_information
        ]
    )

    image_input.change(
        fn=identify_leaf,
        inputs=image_input,
        outputs=[
            prediction_output,
            confidence_output,
            plant_information
        ],
        show_progress="hidden"
    )


if __name__ == "__main__":
    demo.launch()
