# ingest/calls.py
import gc
from pathlib import Path

from faster_whisper import WhisperModel

_model = None


def _get_model():
    """Load the biggest Whisper model that fits in memory, falling back to smaller ones."""
    global _model
    if _model is None:
        for size in ("small", "base", "tiny"):
            try:
                _model = WhisperModel(size, device="cpu", compute_type="int8",
                                      cpu_threads=4, num_workers=1)
                print(f"whisper model: {size}")
                break
            except RuntimeError as e:
                print(f"whisper '{size}' failed to load ({e}); trying a smaller one")
                gc.collect()
        if _model is None:
            raise RuntimeError("No Whisper model fits in memory. Close other apps and retry.")
    return _model


def release_model():
    """Free Whisper's memory before loading the image/text encoders."""
    global _model
    _model = None
    gc.collect()


def ingest_call(path, db, window=60.0):
    """Transcribe, then group segments into ~`window`-second chunks with timestamps.
    Your calls are ~50s, so each becomes ONE chunk and the Q&A stays together."""
    path = Path(path)
    segments, _ = _get_model().transcribe(str(path), vad_filter=True)

    buf, t0, last_end, n = [], None, 0.0, 0
    for s in segments:
        if t0 is None:
            t0 = s.start
        buf.append(s.text.strip())
        last_end = s.end
        if last_end - t0 >= window:
            db.insert_text("audio", path.name, None, " ".join(buf), t_start=t0, t_end=last_end)
            buf, t0, n = [], None, n + 1
    if buf:
        db.insert_text("audio", path.name, None, " ".join(buf), t_start=t0, t_end=last_end)
        n += 1
    print(f"{path.name}: {n} chunk(s)")