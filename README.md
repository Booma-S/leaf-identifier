# Leaf Identifier

## Project Overview

Leaf Identifier is a Digital Image Processing (DIP) and deep learning project that identifies plant species from leaf images and displays available plant information through a Gradio web application.

The project includes image segmentation and preprocessing experiments, dataset preparation, model training and evaluation, error analysis, and a plant-information knowledge base.

## Objectives

- Segment leaf regions using image-processing techniques.
- Audit and prepare a labelled leaf-image dataset.
- Train and compare a baseline CNN with MobileNetV2 transfer learning.
- Evaluate classification performance and inspect prediction errors.
- Provide a simple interface for image-based predictions and plant information.
- Test the application on real-world leaf photographs.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV, NumPy and Pandas
- Matplotlib
- Gradio
- Git and GitHub

## Dataset

The dataset contains **1,835 images across 30 species**.

| Split | Images |
|---|---:|
| Training | 1,283 |
| Validation | 276 |
| Testing | 276 |
| **Total** | **1,835** |

Dataset source: [Mendeley Data – Medicinal Leaf Dataset](https://data.mendeley.com/datasets/nnytj2v3n5/1)

The dataset was audited, and the split files were validated for class coverage and path overlap. An exact-duplicate check found no exact duplicate image contents across splits. Near-duplicate images have not been ruled out.

## Image Processing

Implemented work includes:

- Exploratory data analysis (class distribution, sample images and average colour).
- Otsu-based leaf segmentation using saturation/grayscale thresholding.
- Morphological opening and closing, largest-component selection and hole filling.
- Comparison of 3×3 and 5×5 morphology kernels.
- CLAHE experiments.

Segmentation and enhancement experiments are documented under `reports/`.

## Models and Results

A baseline CNN and a pretrained MobileNetV2 model were trained and evaluated on the held-out test split.

| Metric | Baseline CNN | MobileNetV2 |
|---|---:|---:|
| Test accuracy | 52.17% | 97.83% |
| Macro F1-score | 0.47 | 0.9733 |
| Weighted F1-score | 0.49 | 0.9773 |
| Test images | 276 | 276 |

MobileNetV2 produced the stronger result in the reported evaluation. The error-analysis run identified six incorrect predictions among 276 test images.

## Gradio Application

The application is located at `app/app.py`. It supports:

- Uploading a leaf image.
- Displaying the predicted species.
- Showing the top-five predictions and confidence scores.
- Displaying available information from `knowledge/plants.json`.

The knowledge-base structure exists for all 30 dataset classes, but the plant details and reliable references are still being completed.

## Project Structure

```text
Leaf Identifier/
├── app/
│   └── app.py
├── data/
│   ├── raw/          # Original dataset
│   ├── own/          # Own photos for real-world testing
│   ├── processed/    # Generated segmentation masks
│   └── splits/       # Train/validation/test CSV files
├── knowledge/
│   └── plants.json
├── models/
│   ├── best_leaf_cnn.keras
│   ├── best_mobilenet.keras
│   ├── class_names.json
│   ├── mobilenet_class_names.json
│   └── training histories
├── reports/
│   ├── cnn_baseline/
│   ├── figures/
│   ├── gradcam/
│   ├── mobilenet/
│   ├── segmentation/
│   └── tables/
├── src/
│   ├── audit_dataset.py
│   ├── eda.py
│   ├── segment.py
│   ├── split.py
│   ├── validate_splits.py
│   ├── check_duplicates.py
│   ├── train_cnn.py
│   ├── evaluate_cnn.py
│   ├── train_mobilenet.py
│   ├── evaluate_mobilenet.py
│   ├── analyze_errors.py
│   ├── gradcam.py
│   └── supporting experiment scripts
├── .gitignore
├── requirements.txt
└── README.md
```

*This is a summary of the main folders and scripts; some generated files may be omitted.*

## Setup

Use Python 3.12 if that matches your tested environment.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Make sure the dataset, saved model files and class-name JSON files are present in the paths expected by the scripts. Raw dataset images and large model files may not be included in the Git repository.

## Run the Application

From the project root, run:

```bash
python app/app.py
```

Open the local URL printed by Gradio in the terminal.

## Run Individual Scripts

Run scripts from the project root, and only when you need the corresponding task. Examples:

```bash
# Recreate dataset audit
python src/audit_dataset.py

# Generate EDA outputs
python src/eda.py

# Create segmentation masks
python src/segment.py

# Recreate dataset splits
python src/split.py

# Validate split integrity
python src/validate_splits.py

# Check exact duplicates
python src/check_duplicates.py

# Evaluate the saved MobileNetV2 model
python src/evaluate_mobilenet.py

# Generate detailed test error analysis
python src/analyze_errors.py
```

Training scripts (`src/train_cnn.py` and `src/train_mobilenet.py`) should only be run when you intend to train or retrain a model. Recreating splits or training can overwrite generated outputs, so check the script configuration before running them.

## Remaining Work

- Inspect the six misclassified test images.
- Improve application presentation and uncertainty/error messaging.
- Complete the plant knowledge base with reliable sources.
- Test with own leaf photographs and record the results.
- Finish project documentation, screenshots and final repository checks.
- Review and document the Grad-CAM explainability outputs.

## Limitations

The classifier predicts among the 30 classes in its training dataset. It may make incorrect predictions for unfamiliar species or images taken under different conditions. A confidence score is not a guarantee of correctness.

Plant information is provided for educational purposes and should not be treated as medical advice.
