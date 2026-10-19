# eval/calibrate_threshold.py
"""How well does the top reranker score separate answerable from unanswerable questions?
Runs retrieval only (no API calls)."""
import json
import statistics
from pathlib import Path

from ingest.db import DB
from ingest.encoders import warmup
from retrieval.rerank import retrieve_preset

PRESET = "balanced"

OFF_TOPIC = [
    "what is today's date?",
    "who won the world cup",
    "how do I bake sourdough bread",
    "what is the capital of France",
    "write a poem about autumn",
    "how do I change a car tire",
    "what is the stock price of Apple",
    "how many moons does Jupiter have",
    "translate hello into Spanish",
    "what is the best programming language",
]
# HVAC questions the corpus cannot answer (other brands, other products): the hard abstention case
HVAC_ABSENT = [
    "What is the SEER2 rating of the Lennox XC25 air conditioner?",
    "How do I replace the compressor on a Trane XR14?",
    "What is the warranty on a Rheem tankless water heater?",
    "How do I wire a Honeywell T6 Pro thermostat?",
    "What refrigerant charge does a Goodman GSX14 take?",
    "How do I descale a boiler heat exchanger?",
]


def top_score(db, q):
    hits = retrieve_preset(db, q, 20, PRESET)
    return hits[0].get("rerank") if hits else None


def main():
    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    in_scope = [q for q in qs if q["type"] != "photo"]     # photo requests bypass the reranker

    db = DB()
    warmup()
    try:
        top_score(db, "warmup")
        s_in = [(top_score(db, q["question"]), q) for q in in_scope]
        s_off = [(top_score(db, q), q) for q in OFF_TOPIC]
        s_abs = [(top_score(db, q), q) for q in HVAC_ABSENT]
    finally:
        db.close()

    def vals(rows):
        return [s for s, _ in rows if s is not None]

    v_in, v_off, v_abs = vals(s_in), vals(s_off), vals(s_abs)
    print(f"preset: {PRESET}   (top-1 reranker score; higher = more relevant)\n")
    print(f"in-scope eval questions (n={len(v_in)}): min {min(v_in):.2f}  median {statistics.median(v_in):.2f}  "
          f"max {max(v_in):.2f}")
    print(f"off-topic questions     (n={len(v_off)}): min {min(v_off):.2f}  median {statistics.median(v_off):.2f}  "
          f"max {max(v_off):.2f}")
    print(f"HVAC but not in corpus  (n={len(v_abs)}): min {min(v_abs):.2f}  median {statistics.median(v_abs):.2f}  "
          f"max {max(v_abs):.2f}")

    print("\nlowest-scoring in-scope questions (these set how high the cutoff can go):")
    for s, q in sorted((r for r in s_in if r[0] is not None), key=lambda r: r[0])[:5]:
        print(f"  {s:6.2f}  {q['id']}  {q['question'][:70]}")

    print("\ncutoff   in-scope kept   off-topic refused   HVAC-absent refused")
    for t in (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2):
        kept = sum(v >= t for v in v_in) / len(v_in)
        off = sum(v < t for v in v_off) / len(v_off)
        ab = sum(v < t for v in v_abs) / len(v_abs)
        print(f"{t:>6}   {100 * kept:10.0f}%   {100 * off:14.0f}%   {100 * ab:16.0f}%")

    safe = min(v_in) - 0.5
    print(f"\nsuggested cutoff: {safe:.1f}  (keeps every in-scope question with a 0.5 margin)")
    print("Set it with:  $env:FIELDNOTE_MIN_RERANK = \"%.1f\"   then restart the app." % safe)
    if max(v_off) >= min(v_in):
        print("note: off-topic and in-scope scores overlap, so no cutoff separates them cleanly.")
    print("caution: calibrated on the same questions it is judged on; keep the margin and re-check on new questions.")


if __name__ == "__main__":
    main()