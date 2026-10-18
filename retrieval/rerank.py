# retrieval/rerank.py
import os
import statistics
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from sentence_transformers import CrossEncoder

from retrieval.search import PHOTO_INTENT, retrieve_routed, rrf, search_images, search_text

RERANKER_ID = os.environ.get("RERANKER_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=4)
def _model(max_len):
    return CrossEncoder(RERANKER_ID, max_length=max_len)


def _collapse(items, per_page):
    """Keep at most `per_page` candidates per (source, page). Audio has no page and is never collapsed."""
    seen, out = {}, []
    for c in items:
        key = (c["source"], c.get("page"))
        if key[1] is None:
            out.append(c)
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] <= per_page:
            out.append(c)
    return out


def retrieve_reranked(db, q, k=20, pool=40, rerank_top=80, max_len=384, per_page=0):
    """pool        = candidates fetched per space
       rerank_top  = how many fused candidates the cross-encoder scores (0 = skip the reranker)
       max_len     = tokens per (question, passage) pair
       per_page    = keep at most N candidates per manual page before reranking (0 = no collapsing)"""
    if PHOTO_INTENT.search(q):          # photos have no text for a text cross-encoder
        return retrieve_routed(db, q, k)

    with ThreadPoolExecutor(max_workers=2) as ex:
        ft = ex.submit(search_text, db, q, pool)
        fi = ex.submit(search_images, db, q, pool, ["manual_figure", "manual_page"])
        text_hits, image_hits = ft.result(), fi.result()

    fused = rrf([text_hits, image_hits])
    if per_page:
        fused = _collapse(fused, per_page)
    if rerank_top <= 0:
        return fused[:k]

    cand = fused[:rerank_top]
    scorable = [c for c in cand if (c.get("content") or "").strip()]
    unscorable = [c for c in cand if not (c.get("content") or "").strip()]

    pairs = [(q, c["content"][:1500]) for c in scorable]
    scores = _model(max_len).predict(pairs, batch_size=32, show_progress_bar=False) if pairs else []
    for c, s in zip(scorable, scores):
        c["rerank"] = float(s)
    neutral = statistics.median(c["rerank"] for c in scorable) if scorable else 0.0
    for c in unscorable:                # no nearby text: stays in the pool at a neutral score
        c["rerank"] = neutral

    ranked = sorted(scorable + unscorable, key=lambda c: c["rerank"], reverse=True)
    return ranked[:k]

PRESETS = {
    "fast": dict(pool=40, rerank_top=20, max_len=192, per_page=1),
    "balanced": dict(pool=40, rerank_top=40, max_len=256, per_page=2),
    "high_recall": dict(pool=40, rerank_top=80, max_len=384, per_page=0),
}


def retrieve_preset(db, q, k=20, mode="balanced"):
    return retrieve_reranked(db, q, k, **PRESETS[mode])