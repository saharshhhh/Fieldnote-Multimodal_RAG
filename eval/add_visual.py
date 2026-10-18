# eval/add_visual.py
import argparse
import json
import re
from pathlib import Path

from qdrant_client import models

from ingest.db import DB, IMAGE_COLLECTION

DRAFTS = Path("eval/drafts.json")
BAD = re.compile(r"\b(image|picture|photo|figure|diagram|illustrat\w*|depict\w*|pictured|shown|indicated"
                 r"|described|the literature|this (unit|product|model))\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="e.g. data/pages/OM9GFRC-02_p9_f2.png")
    ap.add_argument("--q", required=True, help="stand-alone question; name the product/model")
    ap.add_argument("--a", required=True, help="short answer visible in the figure")
    args = ap.parse_args()

    if BAD.search(args.q):
        print(f"warning: the question mentions the picture itself ({BAD.search(args.q).group(0)!r}); "
              "a technician wouldn't ask it that way. Reword it.")
    path = Path(args.image).as_posix()
    db = DB()
    try:
        pts, _ = db.client.scroll(
            IMAGE_COLLECTION, limit=1, with_payload=True, with_vectors=False,
            scroll_filter=models.Filter(must=[models.FieldCondition(
                key="image_path", match=models.MatchValue(value=path))]))
    finally:
        db.close()
    if not pts:
        raise SystemExit(f"{path} is not in the index. Copy the path from the contact sheet label.")
    p = pts[0].payload

    drafts = json.loads(DRAFTS.read_text(encoding="utf-8"))
    nid = max(int(d["id"][1:]) for d in drafts) + 1
    drafts.append({
        "id": f"d{nid:03d}", "keep": True, "type": "visual", "question": args.q.strip(),
        "gold_answer": args.a.strip(), "source": p["source"], "page": p["page"],
        "image_path": path, "evidence": "hand-written, verified against the image",
    })
    DRAFTS.write_text(json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"added d{nid:03d}: {p['source']} p.{p['page']}")


if __name__ == "__main__":
    main()