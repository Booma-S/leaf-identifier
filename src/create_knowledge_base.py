import json
from pathlib import Path
import pandas as pd

train_csv = Path("data/splits/train.csv")
output_file = Path("knowledge/plants.json")

if not train_csv.exists():
    raise FileNotFoundError(
        "Training CSV not found. Check that data/splits/train.csv exists."
    )

df = pd.read_csv(train_csv)

if "label" not in df.columns:
    raise ValueError(
        f"Expected a 'label' column. Found: {list(df.columns)}"
    )

species_names = sorted(df["label"].dropna().unique())

plants = {}

for species in species_names:
    plants[species] = {
        "common_name": species,
        "scientific_name": "",
        "description": "",
        "medicinal_uses": [],
        "traditional_uses": [],
        "evidence_note": (
            "Research required. Do not present unverified uses as medical facts."
        ),
        "sources": []
    }

output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text(
    json.dumps(plants, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8"
)

print(f"Created knowledge base: {output_file}")
print(f"Number of species: {len(plants)}")
print("\nSpecies included:")
for species in species_names:
    print(f"- {species}")
