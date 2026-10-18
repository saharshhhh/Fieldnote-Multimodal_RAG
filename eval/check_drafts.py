# eval/check_drafts.py
import json
from pathlib import Path

from eval.common import all_text, norm
from ingest.db import DB

drafts = json.loads(Path("eval/drafts.json").read_text(encoding="utf-8"))
db = DB()
try:
    chunks = [r for r in all_text(db) if r["modality"] == "manual_text"]
finally:
    db.close()

for d in drafts:
    if d["type"] != "text":
        continue
    a = norm(d["gold_answer"])
    gold = (d["source"], d["page"])
    if len(a) < 4:
        print(f"{d['id']}  SHORT ANSWER {d['gold_answer']!r}: verify by eye on {gold}")
        continue
    hits = sorted({(c["source"], c["page"]) for c in chunks if a in norm(c["content"])})
    if gold not in hits:
        print(f"{d['id']}  NOT VERBATIM on gold page {gold}: verify by eye ({d['gold_answer']!r})")
    elif len(hits) == 1:
        print(f"{d['id']}  OK, unique to {gold}")
    else:
        print(f"{d['id']}  AMBIGUOUS, answer appears in {len(hits)} places: {hits}")