from pathlib import Path
from PIL import Image

root = Path("data/raw")
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

total = 0
corrupt = []

for folder in sorted(p for p in root.iterdir() if p.is_dir()):
    images = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    total += len(images)

    for path in images:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            corrupt.append(str(path))

    print(f"{folder.name}: {len(images)} images")

print(f"\nTotal images: {total}")
print(f"Unreadable images: {len(corrupt)}")

for path in corrupt:
    print("CHECK:", path)