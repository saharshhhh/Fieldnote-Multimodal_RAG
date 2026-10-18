# retrieval/try_search.py
import sys

from ingest.db import DB
from retrieval.search import retrieve, search_images, search_text

QUERIES = [
    "why does the indoor coil keep freezing",
    "furnace flame sensor",
    "heat pump outdoor unit covered in ice",
    "air conditioner efficiency rating table",
    "smart thermostat wiring",
]


def show(title, rows, n=6):
    print(f"\n  -- {title}")
    for i, r in enumerate(rows[:n], 1):
        if r.get("page"):
            where = f"p.{r['page']}"
        elif r.get("t_start") is not None:
            where = f"@{r['t_start']:.0f}s"
        else:
            where = ""
        snippet = (r.get("content") or r.get("image_path") or "").replace("\n", " ")[:60]
        score = r.get("rrf", r.get("score", 0))
        print(f"  {i}. [{r['modality']:<13}] {r['source'][:34]:<34} {where:<7} {score:.5f}  {snippet}")


def main():
    db = DB()
    try:
        queries = [" ".join(sys.argv[1:])] if sys.argv[1:] else QUERIES
        for q in queries:
            print(f"\n=== {q}")
            show("text space only", search_text(db, q, 6))
            show("image space only", search_images(db, q, 6))
            show("FUSED (RRF)", retrieve(db, q, 10))
    finally:
        db.close()   # prevents the shutdown traceback


if __name__ == "__main__":
    main()