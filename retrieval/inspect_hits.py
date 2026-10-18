# retrieval/inspect_hits.py
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from ingest.db import DB, TEXT_COLLECTION
from retrieval.search import search_images


def grep_corpus(db, keyword):
    """Case-insensitive substring search over every stored text chunk (manuals + calls)."""
    kw, found, offset = keyword.lower(), [], None
    while True:
        points, offset = db.client.scroll(
            TEXT_COLLECTION, limit=200, offset=offset, with_payload=True, with_vectors=False
        )
        for p in points:
            if kw in p.payload["content"].lower():
                found.append(p.payload)
        if offset is None:
            break
    return found


def contact_sheet(rows, out_path, cols=4, size=256):
    n_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * size, n_rows * (size + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows):
        im = Image.open(r["image_path"]).convert("RGB")
        im.thumbnail((size, size))
        x, y = (i % cols) * size, (i // cols) * (size + 20)
        sheet.paste(im, (x, y))
        draw.text((x + 4, y + size + 4),
                  f"{i + 1}. {Path(r['image_path']).name}  {r['score']:.3f}", fill="black")
    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="question to run against the photo index")
    ap.add_argument("--keyword", help="exact phrase to look for in the text corpus")
    ap.add_argument("--n", type=int, default=8, help="how many photos in the contact sheet")
    args = ap.parse_args()

    db = DB()
    try:
        if args.keyword:
            found = grep_corpus(db, args.keyword)
            print(f'\n"{args.keyword}" appears in {len(found)} text chunk(s):')
            for f in found:
                where = f"p.{f['page']}" if f.get("page") else f"@{(f.get('t_start') or 0):.0f}s"
                print(f"  [{f['modality']}] {f['source']} {where}")

        rows = search_images(db, args.query, args.n, modalities=["photo"])
        out = Path("data/contact_sheet.png")
        contact_sheet(rows, out)
        print(f"\nwrote {out}  (open it and judge: do these look like {args.query!r}?)")
    finally:
        db.close()


if __name__ == "__main__":
    main()