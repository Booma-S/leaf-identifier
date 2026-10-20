# Raw Dataset Documentation

## 1. Dataset Information

- **Dataset name:** Medicinal Leaf Dataset
- **Authors:** Roopashree S and Anitha J
- **Platform:** Mendeley Data
- **Version:** 1
- **Publication date:** 22 October 2020
- **DOI:** 10.17632/nnytj2v3n5.1
- **Original dataset URL:** https://data.mendeley.com/datasets/nnytj2v3n5/1

## 2. Dataset Description

The dataset contains images of medicinal plant leaves organized into species-specific folders.

- **Number of classes:** 30
- **Total number of images:** 1,835
- **Unreadable images detected:** 0
- **Image resolution:** To be verified
- **Image formats:** To be recorded from the downloaded files

### Local Dataset Audit

The initial audit was performed using a Python script and Pillow.

| Audit check | Result |
|---|---:|
| Species folders | 30 |
| Total images | 1,835 |
| Unreadable images | 0 |

All 1,835 image files passed the initial image-integrity check. This does not yet confirm that all images are correctly labelled, free of duplicates, or suitable for segmentation.

## 3. Licence and Attribution

- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Licence URL:** https://creativecommons.org/licenses/by/4.0/
- **Attribution required:** Yes
- **Redistribution:** Subject to the CC BY 4.0 terms and applicable rights.

### Dataset Citation

Roopashree, S., & Anitha, J. (2020). *Medicinal Leaf Dataset* (Version 1). Mendeley Data. https://doi.org/10.17632/nnytj2v3n5.1

### Attribution Notice

The original dataset is credited to Roopashree S and Anitha J and is available through Mendeley Data under the Creative Commons Attribution 4.0 International licence.

Original dataset: https://data.mendeley.com/datasets/nnytj2v3n5/1

Licence: https://creativecommons.org/licenses/by/4.0/

## 4. Data Integrity

The files in this directory are the original downloaded dataset.

- Original images must not be overwritten or modified.
- Preprocessing and segmentation outputs will be stored separately.
- Dataset auditing will check class counts, image counts, file integrity, and image dimensions.
- Dataset splits will be stored separately under `data/splits/`.

## 5. Project Usage

This dataset is used for academic and experimental purposes in the Leaf Identifier project, which explores digital image processing, classical machine learning, and deep learning for medicinal plant leaf identification.

Any modifications or derived datasets will be documented separately, with appropriate attribution to the original dataset creators.
