# ingest/review_photos.py
import csv
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PHOTOS = Path("data/photos")
OUT = Path("data/review")
OUT.mkdir(parents=True, exist_ok=True)
COLS, SIZE = 4, 230


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def sheet(files, title, out_path):
    rows = (len(files) + COLS - 1) // COLS
    img = Image.new("RGB", (COLS * SIZE, rows * (SIZE + 22) + 26), "white")
    d = ImageDraw.Draw(img)
    d.text((6, 6), title, fill="black")
    for i, f in enumerate(files):
        try:
            im = Image.open(PHOTOS / f).convert("RGB")
        except Exception:
            continue
        im.thumbnail((SIZE - 6, SIZE - 6))
        x, y = (i % COLS) * SIZE, (i // COLS) * (SIZE + 22) + 26
        img.paste(im, (x + 3, y))
        d.text((x + 4, y + SIZE - 2), Path(f).stem, fill="black")
    img.save(out_path)


groups = defaultdict(list)
with open(PHOTOS / "manifest.csv", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        groups[row["search_term"]].append(row["file"])

for term, files in groups.items():
    for part, start in enumerate(range(0, len(files), 12), 1):
        out = OUT / f"{slug(term)}_{part}.png"
        sheet(files[start:start + 12], f"label: {term}", out)
        print("wrote", out)
print(f"\nopen the images in {OUT}/ and note the photo numbers that DON'T match their label")