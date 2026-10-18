# eval/visual_candidates.py
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from eval.common import scroll_all
from ingest.db import DB, IMAGE_COLLECTION

OUT = Path("data/visual_candidates")
OUT.mkdir(parents=True, exist_ok=True)
COLS, ROWS, SIZE = 3, 4, 320


def main():
    db = DB()
    try:
        figs = [r for r in scroll_all(db, IMAGE_COLLECTION, ["manual_figure"])
                if not r["source"].startswith("01-")                      # owner manuals only
                and r.get("bbox") and (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]) > 40_000]
    finally:
        db.close()

    by_src = defaultdict(list)
    for r in figs:
        by_src[r["source"]].append(r)

    for src, rows in sorted(by_src.items()):
        rows.sort(key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0]))
        for part, start in enumerate(range(0, len(rows), COLS * ROWS), 1):
            chunk = rows[start:start + COLS * ROWS]
            sheet = Image.new("RGB", (COLS * SIZE, ((len(chunk) + COLS - 1) // COLS) * (SIZE + 24)), "white")
            d = ImageDraw.Draw(sheet)
            for i, r in enumerate(chunk):
                im = Image.open(r["image_path"]).convert("RGB")
                im.thumbnail((SIZE - 8, SIZE - 8))
                x, y = (i % COLS) * SIZE, (i // COLS) * (SIZE + 24)
                sheet.paste(im, (x + 4, y))
                d.text((x + 4, y + SIZE - 4), f"{Path(r['image_path']).stem}  (p.{r['page']})", fill="black")
            out = OUT / f"{Path(src).stem}_{part}.png"
            sheet.save(out)
            print("wrote", out)
    print(f"\n{len(figs)} figures across {len(by_src)} manuals. Open the sheets and look for figures "
          "with labeled parts, indicator icons, terminals or wire colors.")


if __name__ == "__main__":
    main()