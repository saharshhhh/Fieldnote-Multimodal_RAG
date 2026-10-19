# app/runtime.py
"""Process-wide singletons.

Kept in their own module so Streamlit's script reruns (and hot reloads of main.py) never
open the Qdrant local-mode database twice: it allows only one open client per folder.
"""
import threading

_lock = threading.Lock()
_db = None
_booted = False


def get_db():
    global _db
    with _lock:
        if _db is None:
            from ingest.db import DB
            _db = DB()
        return _db


def boot():
    """Open the index and load the models once, so the first question is not slow."""
    global _booted
    db = get_db()
    if not _booted:
        from ingest.encoders import warmup
        from retrieval.rerank import retrieve_preset

        warmup()
        retrieve_preset(db, "warmup", 5, "balanced")   # also loads the cross-encoder
        _booted = True
    return db