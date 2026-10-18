# eval/latency_view.py
import json
import statistics
from pathlib import Path

A = json.loads(Path("eval/answers_v2.json").read_text(encoding="utf-8"))
MAX_TOK = 400

for system in dict.fromkeys(a["system"] for a in A.values()):
    rows = [a for a in A.values() if a["system"] == system and a["out_tokens"] is not None]
    short = sorted(a["llm_ms"] for a in rows if a["out_tokens"] <= MAX_TOK)
    if not short:
        continue
    p95 = short[min(len(short) - 1, int(0.95 * len(short)))]
    ms_per_tok = statistics.median(a["llm_ms"] / max(a["out_tokens"], 1) for a in rows)
    print(f"{system:<18} calls with <= {MAX_TOK} output tokens: {len(short):>2}   "
          f"median LLM ms {statistics.median(short):6.0f}   p95 {p95:6.0f}   "
          f"median ms/token {ms_per_tok:5.1f}")