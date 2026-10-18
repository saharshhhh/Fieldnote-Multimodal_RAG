# eval/inspect_answers.py
import json
import re
from pathlib import Path

ANS = json.loads(Path("eval/answers.json").read_text(encoding="utf-8"))
JUD = json.loads(Path("eval/judgments.json").read_text(encoding="utf-8"))
QS = {}
for line in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines():
    if line.strip():
        q = json.loads(line)
        QS[q["id"]] = q

LEAK = re.compile(r"<[A-Z_]+>|^\s*(the user (asks|is asking)|the question asks|we need to|i need to|let me|okay)", re.I)
CTX_SOURCES = {"fieldnote_lean": 5}      # how many sources that system actually showed the model


def in_context(a, gold, k):
    for label in a["top_sources"][:k]:
        for g in gold:
            if g["source"] not in label:
                continue
            if g.get("page") is None or re.search(rf"\bp\.{g['page']}\b", label):
                return True
    return False


first_leak = None
for system in dict.fromkeys(a["system"] for a in ANS.values()):
    rows = [a for a in ANS.values() if a["system"] == system]
    k = CTX_SOURCES.get(system, 8)
    leaks = [a for a in rows if LEAK.search(a["answer"])]
    unjudged = sorted(a["qid"] for a in rows if f"{system}|{a['qid']}" not in JUD)
    avg_len = sum(len(a["answer"]) for a in rows) / len(rows)
    print(f"\n=== {system}: {len(rows)} answers, avg length {avg_len:.0f} chars")
    print(f"  leaked reasoning in answer: {len(leaks)}")
    for lk in leaks:
        v = JUD.get(f"{system}|{lk['qid']}")
        print(f"    {lk['qid']}: judged {'correct' if v and v['correct'] else 'wrong' if v else 'NO VERDICT'}")
        if first_leak is None and system == "caption_baseline":
            first_leak = lk
    print(f"  no judge verdict: {unjudged}")

    wrong = [a for a in rows if f"{system}|{a['qid']}" in JUD and not JUD[f"{system}|{a['qid']}"]["correct"]]
    ctx_missing = [a for a in wrong if not in_context(a, QS[a["qid"]]["gold_sources"], k)]
    print(f"  wrong answers: {len(wrong)}; of those, gold evidence NOT in the top {k} sources: {len(ctx_missing)}")
    for a in wrong:
        miss = a in ctx_missing
        lk = bool(LEAK.search(a["answer"]))
        print(f"    {a['qid']} [{a['type']:<6}] gold in context: {'NO ' if miss else 'yes'}  "
              f"leak: {'yes' if lk else 'no '}  {a['answer'].replace(chr(10), ' ')[:90]}")

if first_leak:
    print("\n=== full text of one leaked baseline answer "
          f"({first_leak['qid']}, {len(first_leak['answer'])} chars) ===")
    print(first_leak["answer"][:1800])