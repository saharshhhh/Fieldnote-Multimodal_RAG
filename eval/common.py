# eval/common.py
import re

from qdrant_client import models

from ingest.db import IMAGE_COLLECTION, TEXT_COLLECTION


def scroll_all(db, collection, modalities=None):
    flt = None
    if modalities:
        flt = models.Filter(must=[models.FieldCondition(
            key="modality", match=models.MatchAny(any=list(modalities)))])
    out, offset = [], None
    while True:
        points, offset = db.client.scroll(
            collection, scroll_filter=flt, limit=256, offset=offset,
            with_payload=True, with_vectors=False)
        out.extend(p.payload for p in points)
        if offset is None:
            return out


def all_text(db):
    return scroll_all(db, TEXT_COLLECTION)


def norm(s: str) -> str:
    """Lowercase, keep letters/digits/dots, collapse everything else to single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.]+", " ", (s or "").lower())).strip()


__all__ = ["scroll_all", "all_text", "norm", "IMAGE_COLLECTION", "TEXT_COLLECTION"]