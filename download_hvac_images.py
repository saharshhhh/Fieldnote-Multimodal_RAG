
from datasets import load_dataset
from pathlib import Path
import random
import csv

# Output folders
OUT = Path("data/photos")
OUT.mkdir(parents=True, exist_ok=True)

# Load HVAC/electrical manual page images
ds = load_dataset(
    "truskovskiyk/electrical-data-sources",
    split="train"
)

print(f"Dataset contains {len(ds)} pages")
print("Available columns:", ds.column_names)

# Prioritize pages likely to contain useful technical figures
keywords = [
    "wiring", "circuit", "diagram", "terminal",
    "compressor", "electrical", "performance",
    "troubleshooting", "refrigerant cycle",
    "parts", "pressure", "capacity", "connection"
]

candidates = []

for i, row in enumerate(ds):
    text = (row.get("page_text") or "").lower()

    if any(word in text for word in keywords):
        candidates.append(i)

# Randomly choose up to 200 matching pages
random.seed(42)
random.shuffle(candidates)
selected = candidates[:200]

metadata_path = OUT / "metadata.csv"

with metadata_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name", "pdf_name", "pdf_page",
            "page_text", "source_dataset"
        ]
    )
    writer.writeheader()

    for count, idx in enumerate(selected, start=1):
        row = ds[idx]
        image = row["image"]

        filename = f"hvac_page_{count:03d}.png"
        image.convert("RGB").save(OUT / filename)

        writer.writerow({
            "file_name": filename,
            "pdf_name": row.get("pdf_name", ""),
            "pdf_page": row.get("pdf_page", ""),
            "page_text": row.get("page_text", ""),
            "source_dataset":
                "truskovskiyk/electrical-data-sources"
        })

        if count % 25 == 0:
            print(f"Saved {count}/{len(selected)}")

print(f"Done. Images saved to: {OUT.resolve()}")
print(f"Metadata saved to: {metadata_path.resolve()}")
