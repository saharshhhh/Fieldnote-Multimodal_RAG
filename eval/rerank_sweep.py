# eval/rerank_sweep.py
import json
import time
from pathlib import Path

from eval.score import first_rank
from ingest.db import DB
from ingest.encoders import warmup
from retrieval.rerank import retrieve_reranked

K = 20
# (label, pool per space, candidates reranked, tokens per pair, max candidates per page)
CONFIGS = [
    ("current 40/80/384", 40, 80, 384, 0),
    ("40/40/384", 40, 40, 384, 0),
    ("40/30/256", 40, 30, 256, 0),
    ("collapse2 40/40/256", 40, 40, 256, 2),
    ("collapse2 40/30/256", 40, 30, 256, 2),
    ("collapse1 40/30/256", 40, 30, 256, 1),
    ("collapse2 40/20/256", 40, 20, 256, 2),
    ("collapse1 40/20/192", 40, 20, 192, 1),
    ("collapse2 60/30/256", 60, 30, 256, 2),
]
HARD = {"visual", "photo"}


def main():
    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    db = DB()
    warmup()
    print("config                 R@1    R@5   R@20    MRR  hardMRR | p50 ms  p95 ms | q32  q36")
    try:
        for label, pool, top, ml, pp in CONFIGS:
            retrieve_reranked(db, "warmup query", K, pool, top, ml, pp)
            ranks, lat = {}, []
            for q in qs:
                t0 = time.perf_counter()
                items = retrieve_reranked(db, q["question"], K, pool, top, ml, pp)
                lat.append((time.perf_counter() - t0) * 1000)
                ranks[q["id"]] = first_rank(items, q["gold_sources"])

            def rate(k, sel):
                return sum(1 for q in sel if ranks[q["id"]] and ranks[q["id"]] <= k) / len(sel)

            def mrr(sel):
                return sum(1 / ranks[q["id"]] for q in sel if ranks[q["id"]]) / len(sel)

            hard = [q for q in qs if q["type"] in HARD]
            lat.sort()
            p50, p95 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(0.95 * len(lat)))]
            show = lambda r: "-" if r is None else str(r)
            print(f"{label:<21} {100 * rate(1, qs):5.1f}% {100 * rate(5, qs):5.1f}% {100 * rate(20, qs):5.1f}% "
                  f"{mrr(qs):6.3f}  {mrr(hard):6.3f} | {p50:6.0f}  {p95:6.0f} | "
                  f"{show(ranks.get('q32')):>3}  {show(ranks.get('q36')):>3}")
    finally:
        db.close()


if __name__ == "__main__":
    main()