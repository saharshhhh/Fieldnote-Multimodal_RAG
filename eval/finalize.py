# eval/finalize.py
import json
from pathlib import Path

from eval.common import all_text, norm
from ingest.db import DB

AUTO = Path("eval/auto_questions.jsonl")
DRAFTS = Path("eval/drafts.json")
OUT = Path("eval/questions.jsonl")


def main():
    qs = [json.loads(l) for l in AUTO.read_text(encoding="utf-8").splitlines() if l.strip()]
    for d in json.loads(DRAFTS.read_text(encoding="utf-8")):
        if not d.get("keep"):
            continue
        gold = [{"source": d["source"], "page": d["page"]}] + d.get("also", [])
        qs.append({
            "type": d["type"], "question": d["question"], "gold_answer": d["gold_answer"],
            "gold_sources": gold,
        })

    # TEXT-LEAK TEST: a "visual" question is only hard if its answer is NOT in any text chunk
    db = DB()
    try:
        corpus = norm(" ".join(r["content"] for r in all_text(db)))
    finally:
        db.close()
    for q in qs:
        q["text_leak"] = False
        if q["type"] == "visual":
            a = norm(q["gold_answer"])
            if len(a) >= 4 and a in corpus:
                q["text_leak"] = True

    for i, q in enumerate(qs, 1):
        q["id"] = f"q{i:02d}"
    OUT.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")

    counts = {t: sum(q["type"] == t for q in qs) for t in ("text", "audio", "visual", "photo")}
    hard_ok = counts["photo"] + sum(q["type"] == "visual" and not q["text_leak"] for q in qs)
    print(f"wrote {OUT}: {len(qs)} questions {counts}")
    print(f"hard questions (image-only evidence): {hard_ok} (target 15)")
    for q in qs:
        if q["text_leak"]:
            print(f"  LEAK {q['id']}: answer {q['gold_answer']!r} also appears in the PDF text layer -> "
                  f"reclassify as 'text' or rewrite")
    if len(qs) != 40:
        print(f"note: {len(qs)} questions, target is 40")


if __name__ == "__main__":
    main()