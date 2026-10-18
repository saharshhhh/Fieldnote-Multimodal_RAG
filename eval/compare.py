# eval/compare.py
import json
import math
import sys
from pathlib import Path

A = sys.argv[1] if len(sys.argv) > 1 else "caption_baseline"
B = sys.argv[2] if len(sys.argv) > 2 else "fieldnote_routed"

J = json.loads(Path("eval/judgments_v2.json").read_text(encoding="utf-8"))
QS = {}
for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        q = json.loads(line)
        QS[q["id"]] = q


def sign_test(b_only, a_only):
    """Exact two-sided sign test on discordant pairs (null: each is equally likely to win)."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


for name, types in (("visual", {"visual"}), ("ALL", {"text", "audio", "visual"})):
    ids = [i for i, q in QS.items() if q["type"] in types and f"{A}|{i}" in J and f"{B}|{i}" in J]
    b_only = [i for i in ids if J[f"{B}|{i}"]["correct"] and not J[f"{A}|{i}"]["correct"]]
    a_only = [i for i in ids if J[f"{A}|{i}"]["correct"] and not J[f"{B}|{i}"]["correct"]]
    p = sign_test(len(b_only), len(a_only))
    print(f"{name:<7} n={len(ids):<3} only {B} right: {len(b_only)} {b_only}")
    print(f"{'':<7}       only {A} right: {len(a_only)} {a_only}")
    print(f"{'':<7}       sign-test p = {p:.3f}\n")
