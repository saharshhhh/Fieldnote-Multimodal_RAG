# eval/draft_questions.py
import argparse
import base64
import json
import mimetypes
import os
import random
import re
import time
import sys
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
import cohere

load_dotenv()

from eval.common import scroll_all
from ingest.db import DB, IMAGE_COLLECTION, TEXT_COLLECTION

co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
MODEL = os.environ.get("COHERE_MODEL", "command-a-plus-05-2026")
OUT = Path("eval/drafts.json")
MIN_GAP = 3.5          # seconds between calls: Trial key = 20 calls/min

TEXT_PROMPT = """You write evaluation questions for a retrieval system used by HVAC field technicians.
Below is ONE passage from Carrier product literature. Write ONE question this passage answers.
Rules:
- Write in English.
- The question must stand completely alone: include the exact model name, series or product name from the passage.
  Never write "the unit", "the product", "described", "the literature", "the passage" or "the text".
- The answer must be short and factual (a value, a name, a step, a clearance, a time).
- Return {{"skip": true}} for warranty, legal, history, contact-information, marketing-fluff, or boilerplate passages,
  and when the passage has no concrete technical fact tied to a named product.
Return JSON only: {{"question": "...", "answer": "..."}}

PASSAGE:
{passage}"""

VISUAL_PROMPT = """You write evaluation questions for a retrieval system used by HVAC field technicians.
The attached image is a figure, table, or diagram cropped from Carrier product literature. Write ONE question
whose answer can only be found by LOOKING at this image (a value in a table or chart, which part a label points
to, which terminal or wire, a visible state or shape).
Rules:
- Write in English.
- Stand-alone: include the product, series or topic so the question is unambiguous.
  Never write "this image", "the figure", "the diagram" or "the product".
- The answer must be short and clearly visible in the image. Do not guess.
- Return {{"skip": true}} if the image has no readable fact (decorative photo, logo, unreadable).
Return JSON only: {{"question": "...", "answer": "..."}}"""

BAD_QUESTION = re.compile(
    r"\b(described|the literature|the passage|the text|the document"
    r"|this (unit|product|model)|the (product|unit)"
    r"|image|picture|photo|figure|diagram|illustrat\w*|depict\w*|pictured|shown|indicated)\b",
    re.I)
EN = {"the", "and", "to", "of", "is", "in", "for", "with", "that", "this", "are", "be", "or", "on"}
FOREIGN = {"el", "la", "de", "los", "las", "del", "para", "con", "une", "les", "des", "pour",
           "est", "et", "du", "que", "por", "una"}

_last_call = 0.0


def throttle():
    global _last_call
    wait = MIN_GAP - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def looks_english(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    if len(words) < 30:
        return False
    return sum(w in EN for w in words) > 2 * sum(w in FOREIGN for w in words)


def data_url(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.standard_b64encode(Path(path).read_bytes()).decode()


def ask(content):
    use_format = True
    for attempt in range(5):
        throttle()
        kwargs = dict(model=MODEL, max_tokens=300, temperature=0.3,
                      messages=[{"role": "user", "content": content}])
        if use_format:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = co.chat(**kwargs)
            break
        except Exception as e:
            if "429" in str(e):
                body = str(getattr(e, "body", e))
                if "month" in body.lower():
                    sys.exit("Monthly Trial-key cap reached. Drafts so far are saved in "
                             f"{OUT}. Message: {body[:200]}")
                wait = 20 + 10 * attempt
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
            elif use_format:
                use_format = False          # retry once without JSON mode
            else:
                raise
    else:
        sys.exit(f"Still rate limited after 5 tries. Is another script using this key? "
                 f"Drafts so far are saved in {OUT}.")
    text = "".join(b.text for b in resp.message.content if getattr(b, "type", None) == "text")
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"skip": True}

def round_robin(rows, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
    for g in groups.values():
        rng.shuffle(g)
    order = []
    while any(groups.values()):
        for src in sorted(groups):
            if groups[src]:
                order.append(groups[src].pop())
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=int, default=24)
    ap.add_argument("--visual", type=int, default=18)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--append", action="store_true",
                    help="add to the existing drafts.json instead of replacing it")
    args = ap.parse_args()

    drafts = []
    if args.append and OUT.exists():
        drafts = json.loads(OUT.read_text(encoding="utf-8"))
    used_images = {d["image_path"] for d in drafts if d.get("image_path")}
    used_pages = {(d["source"], d["page"]) for d in drafts if d["type"] == "text"}
    counter = [max([int(d["id"][1:]) for d in drafts] + [0]) + 1]

    db = DB()
    try:
        chunks, figs = [], []
        if args.text > 0:
            chunks = [r for r in scroll_all(db, TEXT_COLLECTION, ["manual_text"])
                      if len(r["content"]) > 350 and looks_english(r["content"])
                      and (r["source"], r["page"]) not in used_pages]
        if args.visual > 0:
            for r in scroll_all(db, IMAGE_COLLECTION, ["manual_figure"]):
                b = r.get("bbox")
                if b and (b[2] - b[0]) * (b[3] - b[1]) > 40_000 and r["image_path"] not in used_images:
                    figs.append(r)
    finally:
        db.close()

    # owner manuals (wiring, indicators, assembly drawings) first, marketing brochures after
    owner = [r for r in figs if not r["source"].startswith("01-")]
    brochure = [r for r in figs if r["source"].startswith("01-")]
    fig_order = round_robin(owner, args.seed + 1) + round_robin(brochure, args.seed + 1)

    print(f"{len(chunks)} usable English text chunks, {len(figs)} usable figure crops "
          f"({len(owner)} from owner manuals)")
    print(f"(~{(args.text + args.visual) * MIN_GAP / 60:.0f}+ minutes at the Trial-key rate limit)\n")

    def save():
        OUT.write_text(json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8")

    def accept(kind, r, d, evidence):
        q = (d.get("question") or "").strip()
        if d.get("skip") or not q or BAD_QUESTION.search(q):
            return False
        drafts.append({
            "id": f"d{counter[0]:03d}", "keep": False, "type": kind,
            "question": q, "gold_answer": str(d.get("answer", "")).strip(),
            "source": r["source"], "page": r.get("page"),
            "image_path": r.get("image_path"), "evidence": evidence,
        })
        counter[0] += 1
        save()                      # incremental: a crash never loses drafts
        print(f"  [{kind}] {q[:95]}")
        return True

    got = 0
    for r in round_robin(chunks, args.seed):
        if got >= args.text:
            break
        try:
            d = ask([{"type": "text", "text": TEXT_PROMPT.format(passage=r["content"][:1800])}])
        except Exception as e:
            print("  error:", str(e)[:120])
            continue
        got += accept("text", r, d, r["content"][:300].replace("\n", " "))

    got = 0
    for r in fig_order:
        if got >= args.visual:
            break
        try:
            d = ask([{"type": "text", "text": VISUAL_PROMPT},
                     {"type": "image_url", "image_url": {"url": data_url(r["image_path"]), "detail": "high"}}])
        except Exception as e:
            print("  error:", str(e)[:120])
            continue
        got += accept("visual", r, d, (r.get("context_text") or "")[:200].replace("\n", " "))

    print(f"\nwrote {OUT}: {len(drafts)} drafts. Verify each, set \"keep\": true on the good ones.")


if __name__ == "__main__":
    main()