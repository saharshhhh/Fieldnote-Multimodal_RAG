# ingest/run_all.py
import argparse
from pathlib import Path

from ingest.calls import ingest_call, release_model
from ingest.db import DB
from ingest.encoders import warmup
from ingest.manuals import ingest_manual
from ingest.photos import ingest_photos

DATA = Path("data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="empty the collections first")
    ap.add_argument("--only", choices=["manuals", "photos", "calls"])
    args = ap.parse_args()

    db = DB()
    if args.reset:
        db.reset()

    # Calls go FIRST: Whisper is loaded and released before SigLIP/bge take memory.
    if args.only in (None, "calls"):
        audio_files = sorted((DATA / "calls").glob("*.mp3"))
        print(f"found {len(audio_files)} audio files")
        for audio in audio_files:
            ingest_call(audio, db)
        release_model()

    if args.only in (None, "manuals", "photos"):
        warmup()

    if args.only in (None, "manuals"):
        for pdf in sorted((DATA / "manuals").glob("*.pdf")):
            ingest_manual(pdf, DATA / "pages", db)
    if args.only in (None, "photos"):
        ingest_photos(DATA / "photos", db)

    print("\nrows per modality:")
    for r in db.counts():
        print(f"  {r['modality']:<14} {r['n']}")
    db.close()


if __name__ == "__main__":
    main()