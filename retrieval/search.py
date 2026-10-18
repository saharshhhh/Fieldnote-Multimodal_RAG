# retrieval/search.py
from concurrent.futures import ThreadPoolExecutor
import re
from ingest.encoders import embed_image_query, embed_text_query


def search_text(db, q, k=20, modalities=None):
    return db.search_text(embed_text_query(q), k, modalities)


def search_images(db, q, k=20, modalities=None):
    return db.search_images(embed_image_query(q), k, modalities)


PHOTO_INTENT = re.compile(r"\b(photos?|pictures?|pics?|images?)\b|\blooks? like\b", re.I)


def rrf(result_lists, k=60, weights=None):
    """Reciprocal Rank Fusion with optional per-list weights.
    Uses rank positions only, because scores from different embedding spaces are not comparable."""
    weights = weights or [1.0] * len(result_lists)
    fused = {}
    for lst, w in zip(result_lists, weights):
        for rank, r in enumerate(lst):
            key = (r["modality"], r["id"])
            fused.setdefault(key, {**r, "rrf": 0.0})["rrf"] += w / (k + rank + 1)
    return sorted(fused.values(), key=lambda r: r["rrf"], reverse=True)


def retrieve(db, q, k=20):
    with ThreadPoolExecutor(max_workers=2) as ex:
        ft = ex.submit(search_text, db, q, k)
        fi = ex.submit(search_images, db, q, k)
        text_hits, image_hits = ft.result(), fi.result()
    return rrf([text_hits, image_hits])[:k]


def retrieve_routed(db, q, k=20):
    """Same as retrieve(), but a question that asks for a photo/picture searches only the photo
    collection and weights it 4x over the text list."""
    wants_photo = bool(PHOTO_INTENT.search(q))
    with ThreadPoolExecutor(max_workers=2) as ex:
        ft = ex.submit(search_text, db, q, k)
        fi = ex.submit(search_images, db, q, k, ["photo"] if wants_photo else None)
        text_hits, image_hits = ft.result(), fi.result()
    weights = [0.5, 2.0] if wants_photo else [1.0, 1.0]
    return rrf([text_hits, image_hits], weights=weights)[:k]