# Leaf Identifier

## Project Overview

Leaf Identifier is a Digital Image Processing (DIP) and machine learning project for identifying plant species from leaf images and displaying relevant plant information.

The project uses image preprocessing, segmentation, morphological operations, feature extraction, and classification. A deep learning model will also be evaluated against classical machine learning approaches.

## Objectives

- Segment leaf regions from images using image processing techniques.
- Extract shape, colour, and texture features.
- Compare classical machine learning models with a CNN.
- Evaluate robustness on real-world leaf photographs.
- Display predictions and plant information through a web application.

## Project Structure

- `data/raw/` — Original leaf dataset.
- `data/own/` — Photographs collected for real-world testing.
- `data/splits/` — Train, validation, and test split files.
- `src/` — Image processing, feature extraction, training, and inference.
- `knowledge/` — Plant information and references.
- `app/` — Web application.
- `reports/` — Experimental results and figures.

## Setup

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```
