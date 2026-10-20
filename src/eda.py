
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Collect images from each species folder
species_data = {}

for folder in sorted(DATA_DIR.iterdir()):
    if not folder.is_dir():
        continue

    images = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if images:
        species_data[folder.name] = images

if not species_data:
    raise RuntimeError(f"No species folders or images found in {DATA_DIR}")

species_names = list(species_data)
image_counts = [len(species_data[name]) for name in species_names]

print(f"Species: {len(species_names)}")
print(f"Images: {sum(image_counts)}")


# Figure 1: Class distribution
fig, ax = plt.subplots(figsize=(13, 9))

y_positions = np.arange(len(species_names))
ax.barh(y_positions, image_counts)
ax.set_yticks(y_positions)
ax.set_yticklabels(species_names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Number of images")
ax.set_title("Leaf Dataset: Image Count per Species")
ax.grid(axis="x", alpha=0.25)

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# Figure 2: Sample image grid (one image per species)
cols = 5
rows = int(np.ceil(len(species_names) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.2))
axes = np.array(axes).reshape(-1)

for ax, species in zip(axes, species_names):
    image_path = species_data[species][0]

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        ax.imshow(image)

    ax.set_title(species, fontsize=8)
    ax.axis("off")

for ax in axes[len(species_names):]:
    ax.axis("off")

fig.suptitle("Representative Images from Each Leaf Species", fontsize=15)
fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "sample_grid.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)


# Figure 3: Average RGB colour per species
# This initial measurement includes the background.
# A later segmentation-based analysis can isolate leaf colours.
mean_colours = []

for species in species_names:
    image_colours = []

    for image_path in species_data[species]:
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image = image.resize((128, 128))
                pixels = np.asarray(image, dtype=np.float32)
                image_colours.append(pixels.mean(axis=(0, 1)))
        except Exception as error:
            print(f"Skipping {image_path}: {error}")

    if not image_colours:
        mean_colours.append([np.nan, np.nan, np.nan])
    else:
        # Equal weight for each image within a species
        mean_colours.append(np.mean(image_colours, axis=0))

mean_colours = np.asarray(mean_colours)

fig, ax = plt.subplots(figsize=(10, 11))

valid = np.isfinite(mean_colours).all(axis=1)
valid_names = np.array(species_names)[valid]
valid_colours = mean_colours[valid]

# RGB values are converted to [0, 1] for displaying swatches
swatches = np.clip(valid_colours / 255.0, 0, 1).reshape(-1, 1, 3)

ax.imshow(swatches, aspect="auto")
ax.set_yticks(np.arange(len(valid_names)))
ax.set_yticklabels(valid_names, fontsize=8)
ax.set_xticks([])
ax.set_title("Mean Image Colour per Species (RGB)")
ax.set_xlabel("Mean RGB colour swatch")

fig.tight_layout()
fig.savefig(
    FIGURES_DIR / "average_colour_per_species.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("\nEDA completed. Figures saved to:")
print(FIGURES_DIR)

for figure in [
    "class_distribution.png",
    "sample_grid.png",
    "average_colour_per_species.png",
]:
    print(f"- {figure}")
