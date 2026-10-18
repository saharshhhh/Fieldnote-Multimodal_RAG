# eval/score.py
import json
import time
from pathlib import Path
from retrieval.rerank import retrieve_preset
from eval.baseline import baseline_retrieve
from ingest.db import DB
from ingest.encoders import warmup
from retrieval.search import retrieve, retrieve_routed, search_images, search_text
K = 20
KS = (1, 5, 10, 20)
HARD = {"visual", "photo"}

SYSTEMS = {
    "caption_baseline": lambda db, q: baseline_retrieve(db, q, K),
    "text_only": lambda db, q: search_text(db, q, K),
    "image_only": lambda db, q: search_images(db, q, K),
    "fieldnote_rrf": lambda db, q: retrieve(db, q, K),
    "fieldnote_routed": lambda db, q: retrieve_routed(db, q, K),
    "rerank_fast": lambda db, q: retrieve_preset(db, q, K, "fast"),
    "rerank_balanced": lambda db, q: retrieve_preset(db, q, K, "balanced"),
    "rerank_high": lambda db, q: retrieve_preset(db, q, K, "high_recall"),
}


def is_hit(item, gold):
    for g in gold:
        if item["source"] != g["source"]:
            continue
        if g.get("page") is None or item.get("page") == g["page"]:
            return True
    return False


def first_rank(items, gold):
    for i, it in enumerate(items, 1):
        if is_hit(it, gold):
            return i
    return None


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    db = DB()
    warmup()
    ranks = {s: {} for s in SYSTEMS}
    lat = {s: [] for s in SYSTEMS}
    try:
        for s, fn in SYSTEMS.items():
            fn(db, "warmup query")                      # keep model load out of the timings
            for q in qs:
                t0 = time.perf_counter()
                items = fn(db, q["question"])
                lat[s].append((time.perf_counter() - t0) * 1000)
                ranks[s][q["id"]] = first_rank(items, q["gold_sources"])
    finally:
        db.close()

    def metrics(system, subset):
        rs = [ranks[system][q["id"]] for q in subset]
        out = {f"R@{k}": sum(1 for r in rs if r is not None and r <= k) / len(rs) for k in KS}
        out["MRR"] = sum(1 / r for r in rs if r is not None) / len(rs)
        return out

    groups = {
        "ALL": qs,
        "EASY (text+audio)": [q for q in qs if q["type"] not in HARD],
        "HARD (visual+photo)": [q for q in qs if q["type"] in HARD],
        "visual": [q for q in qs if q["type"] == "visual"],
        "photo": [q for q in qs if q["type"] == "photo"],
        "text": [q for q in qs if q["type"] == "text"],
        "audio": [q for q in qs if q["type"] == "audio"],
    }
    results = {}
    for gname, subset in groups.items():
        print(f"\n=== {gname}  (n={len(subset)})")
        print(f"{'system':<18}" + "".join(f"{'R@' + str(k):>8}" for k in KS) + f"{'MRR':>8}")
        for s in SYSTEMS:
            m = metrics(s, subset)
            results.setdefault(gname, {})[s] = m
            print(f"{s:<18}" + "".join(f"{pct(m['R@' + str(k)]):>8}" for k in KS) + f"{m['MRR']:>8.3f}")

    print("\n=== latency (retrieval only, ms)")
    latency = {}
    for s in SYSTEMS:
        xs = sorted(lat[s])
        p50 = xs[len(xs) // 2]
        p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
        latency[s] = {"p50": round(p50), "p95": round(p95)}
        print(f"{s:<18} p50 {p50:7.0f}   p95 {p95:7.0f}")

    print("\n=== HARD questions: rank of first correct hit (- = miss in top 20)")
    names = list(SYSTEMS)
    print(f"{'id':<5}{'type':<8}" + "".join(f"{n[:14]:>16}" for n in names) + "  question")
    for q in groups["HARD (visual+photo)"]:
        cells = "".join(f"{(str(ranks[n][q['id']]) if ranks[n][q['id']] else '-'):>16}" for n in names)
        print(f"{q['id']:<5}{q['type']:<8}{cells}  {q['question'][:55]}")

    Path("eval/results.json").write_text(
        json.dumps({"metrics": results, "latency": latency, "ranks": ranks}, indent=2), encoding="utf-8")

    hard, allq = results["HARD (visual+photo)"], results["ALL"]
    md = ["| System | R@20 (15 hard) | MRR (15 hard) | R@5 (all 40) | R@20 (all 40) | p95 ms |",
          "|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        md.append(f"| {s} | {pct(hard[s]['R@20']).strip()} | {hard[s]['MRR']:.3f} | "
                  f"{pct(allq[s]['R@5']).strip()} | {pct(allq[s]['R@20']).strip()} | {latency[s]['p95']} |")
    Path("eval/results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nwrote eval/results.json and eval/results.md (README table)")


if __name__ == "__main__":
    main()