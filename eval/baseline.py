# eval/baseline.py
from qdrant_client import models

from ingest.encoders import embed_text_query

CAPTION_COLLECTION = "caption_chunks"


def ensure_collection(db, recreate=False):
    c = db.client
    if recreate and c.collection_exists(CAPTION_COLLECTION):
        c.delete_collection(CAPTION_COLLECTION)
    if not c.collection_exists(CAPTION_COLLECTION):
        c.create_collection(
            CAPTION_COLLECTION,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))


def baseline_retrieve(db, q, k=20):
    """One text space: manual text + transcripts + image captions, all embedded with bge.
    Scores ARE comparable here (same model), so a plain score sort is a legitimate merge."""
    v = embed_text_query(q)
    rows = db.search_text(v, k)
    res = db.client.query_points(CAPTION_COLLECTION, query=v, limit=k, with_payload=True)
    for p in res.points:
        rows.append({
            "id": p.id, "modality": p.payload["modality"] + "_caption",
            "source": p.payload["source"], "page": p.payload.get("page"), "t_start": None,
            "content": p.payload["content"], "image_path": p.payload["image_path"],
            "score": p.score,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:k]