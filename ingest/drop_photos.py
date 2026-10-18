# ingest/drop_photos.py
import csv
import sys
from pathlib import Path

from qdrant_client import models

from ingest.db import DB, IMAGE_COLLECTION

PHOTOS = Path("data/photos")
stems = {a.removesuffix(".jpg").removesuffix(".png") for a in sys.argv[1:]}
if not stems:
    sys.exit("usage: python -m ingest.drop_photos photo_012 photo_047 ...")

# 1) files
for p in PHOTOS.glob("photo_*"):
    if p.stem in stems:
        p.unlink()
        print("deleted", p.name)

# 2) manifest rows
manifest = PHOTOS / "manifest.csv"
with open(manifest, encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
keep = [r for r in rows if Path(r["file"]).stem not in stems]
with open(manifest, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(keep)
print(f"manifest: {len(rows)} -> {len(keep)} rows")

# 3) index entries (no-op if not ingested yet)
db = DB()
try:
    for r in rows:
        if Path(r["file"]).stem in stems:
            flt = models.Filter(must=[models.FieldCondition(
                key="image_path", match=models.MatchValue(value=f"data/photos/{r['file']}"))])
            db.client.delete(IMAGE_COLLECTION, points_selector=models.FilterSelector(filter=flt))
finally:
    db.close()