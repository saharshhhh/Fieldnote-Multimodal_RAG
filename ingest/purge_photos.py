# ingest/purge_photos.py
import shutil
from pathlib import Path

from qdrant_client import models

from ingest.db import DB, IMAGE_COLLECTION

db = DB()
try:
    flt = models.Filter(must=[models.FieldCondition(
        key="modality", match=models.MatchValue(value="photo"))])
    before = db.client.count(IMAGE_COLLECTION, count_filter=flt, exact=True).count
    db.client.delete(IMAGE_COLLECTION, points_selector=models.FilterSelector(filter=flt))
    after = db.client.count(IMAGE_COLLECTION, count_filter=flt, exact=True).count
    print(f"photo points in index: {before} -> {after}")
finally:
    db.close()

dest = Path("data/photos_old_code_pages")
dest.mkdir(exist_ok=True)
moved = 0
for p in Path("data/photos").glob("hvac_page_*"):
    shutil.move(str(p), dest / p.name)
    moved += 1
print(f"moved {moved} old code-page files to {dest}")