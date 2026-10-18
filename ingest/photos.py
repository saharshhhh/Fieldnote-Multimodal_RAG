# ingest/photos.py
from pathlib import Path

from PIL import Image

from ingest.encoders import embed_images

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ingest_photos(folder, db, batch=16):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)

    good = []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()          # catches truncated/corrupt downloads
            good.append(p)
        except Exception as e:
            print(f"skipping {p.name}: {e}")

    for i in range(0, len(good), batch):
        chunk = good[i:i + batch]
        vecs = embed_images([str(p) for p in chunk])
        for p, v in zip(chunk, vecs):
            db.insert_image("photo", p.name, None, p.as_posix(),
                            bbox=None, context=None, vec=v)
        print(f"photos: {min(i + batch, len(good))}/{len(good)}")