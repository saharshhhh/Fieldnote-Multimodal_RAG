# eval/build_baseline.py
import json
import uuid
from pathlib import Path

from qdrant_client import models

from eval.baseline import CAPTION_COLLECTION, ensure_collection
from eval.common import scroll_all
from ingest.db import DB, IMAGE_COLLECTION
from ingest.encoders import embed_texts

caps = json.loads(Path("eval/captions.json").read_text(encoding="utf-8"))
db = DB()
try:
    meta = {r["image_path"]: r for r in scroll_all(db, IMAGE_COLLECTION)}
    ensure_collection(db, recreate=True)
    items = [(p, c) for p, c in caps.items() if p in meta and c]
    for i in range(0, len(items), 32):
        batch = items[i:i + 32]
        vecs = embed_texts([c for _, c in batch])
        db.client.upsert(CAPTION_COLLECTION, points=[
            models.PointStruct(
                id=str(uuid.uuid4()), vector=v,
                payload={"modality": meta[p]["modality"], "source": meta[p]["source"],
                         "page": meta[p].get("page"), "image_path": p, "content": c})
            for (p, c), v in zip(batch, vecs)])
        print(f"embedded {min(i + 32, len(items))}/{len(items)}")
    print(f"\nbaseline index ready: {len(items)} caption chunks")
finally:
    db.close()