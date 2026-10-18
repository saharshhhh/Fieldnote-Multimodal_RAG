# eval/caption_all.py
import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import cohere

from eval.common import scroll_all
from ingest.db import DB, IMAGE_COLLECTION

co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
MODEL = os.environ.get("COHERE_MODEL", "command-a-plus-05-2026")
CACHE = Path("eval/captions.json")
MIN_GAP = 3.5            # Trial key: 20 calls/min

PROMPT = """You are indexing an image from HVAC equipment literature or a job-site photo for a TEXT search engine.
Describe it so that someone searching in words can find it. Include: what it shows, every readable label,
number, unit, table heading and value, icon meaning, part names, colors, and any visible condition (ice, rust, burn marks).
Be factual and do not guess. Maximum 150 words. Plain text only."""

_last = 0.0


def throttle():
    global _last
    wait = MIN_GAP - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.standard_b64encode(Path(path).read_bytes()).decode()


THINKING_OFF = True   # ask the model to skip its reasoning pass; auto-disabled if unsupported


def _call(path, max_tokens, thinking_off):
    kwargs = dict(
        model=MODEL, max_tokens=max_tokens, temperature=0.1,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url(path), "detail": "high"}},
        ]}])
    if thinking_off:
        kwargs["thinking"] = {"type": "disabled"}
    return co.chat(**kwargs)


def caption(path):
    global THINKING_OFF
    for attempt in range(5):
        throttle()
        try:
            try:
                resp = _call(path, 1500, THINKING_OFF)
            except TypeError:                       # older SDK without the `thinking` argument
                THINKING_OFF = False
                resp = _call(path, 1500, False)
            blocks = resp.message.content or []
            text = "".join(b.text for b in blocks if getattr(b, "type", None) == "text").strip()
            if text:
                return text
            kinds = [getattr(b, "type", type(b).__name__) for b in blocks]
            print(f"  EMPTY caption. blocks={kinds} finish_reason={getattr(resp, 'finish_reason', None)} "
                  f"usage={getattr(resp, 'usage', None)}")
            return None
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                body = str(getattr(e, "body", e))
                if "month" in body.lower():
                    sys.exit(f"Monthly Trial-key cap reached. Progress is saved in {CACHE}. {body[:200]}")
                wait = 20 + 10 * attempt
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
            elif THINKING_OFF and "thinking" in msg.lower():
                print("  model rejected the 'thinking' option; retrying without it")
                THINKING_OFF = False
            else:
                print("  error:", msg[:160])
                return None
    sys.exit(f"Still rate limited. Is another script using this key? Progress is saved in {CACHE}.")

def save(caps):
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(caps, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, CACHE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="caption at most N new images (0 = all)")
    ap.add_argument("--types", default="photo,manual_figure,manual_page")
    args = ap.parse_args()

    caps = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    db = DB()
    try:
        rows = scroll_all(db, IMAGE_COLLECTION, args.types.split(","))
    finally:
        db.close()

    todo = [r for r in rows if r["image_path"] not in caps and Path(r["image_path"]).exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} images in scope, {len(caps)} already captioned, {len(todo)} to do "
          f"(~{len(todo) * (MIN_GAP + 2) / 60:.0f} min). Safe to Ctrl+C and resume.\n")

    for i, r in enumerate(todo, 1):
        text = caption(r["image_path"])
        if text:
            caps[r["image_path"]] = text
            if i % 5 == 0:
                save(caps)
            print(f"[{i}/{len(todo)}] {Path(r['image_path']).name}: {text[:80]!r}")
    save(caps)
    print(f"\nsaved {len(caps)} captions to {CACHE}")


if __name__ == "__main__":
    main()