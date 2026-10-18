# eval/make_auto_questions.py
import csv
import json
from pathlib import Path

CALLS = Path("data/calls/gold_facts.json")
PHOTOS = Path("data/photos")
OUT = Path("eval/auto_questions.jsonl")

# photo label -> how a technician would phrase the search
PHOTO_QUERIES = {
    "heat pump outdoor unit": "outdoor heat pump unit",
    "humidifier": "room humidifier",
    "ice on heat pump": "ice forming on a heat pump",
    "run capacitor": "motor run capacitor",
    "rooftop unit hvac": "rooftop HVAC unit",
}


def main():
    qs = []

    # audio: first fact from each call -> one question per call
    seen = set()
    for f in json.loads(CALLS.read_text(encoding="utf-8")):
        if f["source"] in seen:
            continue
        seen.add(f["source"])
        qs.append({
            "type": "audio", "question": f["question"], "gold_answer": f["gold_answer"],
            "gold_sources": [{"source": f["source"]}],
        })

    # photos: every surviving verified photo under the label is a correct answer
    by_label = {}
    with open(PHOTOS / "manifest.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if (PHOTOS / row["file"]).exists():
                by_label.setdefault(row["search_term"], []).append(row["file"])
    for label, query in PHOTO_QUERIES.items():
        files = by_label.get(label, [])
        if len(files) < 2:
            print(f"skipping photo label {label!r}: only {len(files)} photo(s) left")
            continue
        qs.append({
            "type": "photo", "question": f"Show me a photo of a {query}",
            "gold_answer": f"any of {len(files)} verified photos",
            "gold_sources": [{"source": f} for f in files],
        })

    OUT.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qs) + "\n", encoding="utf-8")
    n_a = sum(q["type"] == "audio" for q in qs)
    n_p = sum(q["type"] == "photo" for q in qs)
    print(f"wrote {OUT}: {n_a} audio, {n_p} photo questions")


if __name__ == "__main__":
    main()