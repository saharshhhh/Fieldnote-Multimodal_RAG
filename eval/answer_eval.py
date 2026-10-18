# eval/answer_eval.py
import argparse
import base64
import io
import json
import mimetypes
import os
import re
import statistics
import sys
import time
from pathlib import Path

import cohere
from PIL import Image

from eval.baseline import baseline_retrieve
from ingest.db import DB
from ingest.encoders import warmup
from retrieval.search import retrieve_routed

co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
MODEL = os.environ.get("COHERE_MODEL", "command-a-plus-05-2026")
MIN_GAP = 3.5                      # Trial key: 20 calls/min
K_SOURCES, MAX_IMAGES = 8, 4
ANSWERS, JUDGE = Path("eval/answers_v2.json"), Path("eval/judgments_v2.json")

SYSTEM_PROMPT = (
    "You assist HVAC field technicians. Answer ONLY from the numbered sources. "
    "Read values directly from charts, tables and diagrams when relevant. "
    "Think silently. Output ONLY your final answer inside <answer></answer> tags: one or two sentences, "
    "citing sources like [2]. If the sources don't contain the answer, say so inside the tags."
)
SYSTEM_PROMPT_JSON = (
    "You assist HVAC field technicians. Answer ONLY from the numbered sources. "
    "Read values directly from charts, tables and diagrams when relevant. "
    'Respond with JSON only, exactly in this form: {"answer": "<one or two sentences, citing sources like [2]; '
    "if the sources don't contain the answer, say so here>\"}"
)
JUDGE_PROMPT = """You grade answers from an HVAC question-answering system.
Question: {q}
Gold answer: {gold}
System output: {ans}

The system output may begin with the model's scratch reasoning; grade the FINAL answer it commits to
(usually the last statement). It is correct if it states the gold answer's key fact(s); different wording,
digits vs words, and extra detail are fine. It is incorrect if it gives a different fact, contradicts the gold
answer, or says the sources don't contain the answer.
Return JSON only: {{"correct": true or false, "reason": "<=12 words"}}"""

ABSTAIN = re.compile(r"(do(es)? not (contain|include|mention|provide)|cannot (find|answer)|can't (find|answer)"
                     r"|no information|not (provided|mentioned|specified) in the sources)", re.I)
THINKING_OFF = os.environ.get("COHERE_THINKING", "off") == "off"
_last = 0.0


def throttle():
    global _last
    wait = MIN_GAP - (time.monotonic() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.monotonic()


def _out_tokens(resp):
    try:
        return int(resp.usage.tokens.output_tokens)
    except Exception:
        return None


def _truncated(resp):
    fr = str(getattr(resp, "finish_reason", "")).upper()
    return "MAX_TOKENS" in fr or "LENGTH" in fr


def chat(content, system=None, max_tokens=1500, json_mode=False):
    """Returns {"text","ms","out_tokens","truncated"} or None. ms excludes throttle sleeps."""
    global THINKING_OFF
    use_json = json_mode
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    for attempt in range(5):
        throttle()
        kwargs = dict(model=MODEL, max_tokens=max_tokens, temperature=0.1, messages=msgs)
        if THINKING_OFF:
            kwargs["thinking"] = {"type": "disabled"}
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            try:
                resp = co.chat(**kwargs)
            except TypeError:
                THINKING_OFF = False
                kwargs.pop("thinking", None)
                resp = co.chat(**kwargs)
            ms = (time.perf_counter() - t0) * 1000
            text = "".join(b.text for b in resp.message.content if getattr(b, "type", None) == "text")
            return {"text": text.strip(), "ms": ms, "out_tokens": _out_tokens(resp), "truncated": _truncated(resp)}
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                body = str(getattr(e, "body", e))
                if "month" in body.lower():
                    sys.exit(f"Monthly Trial-key cap reached. Progress is saved. {body[:200]}")
                wait = 20 + 10 * attempt
                print(f"  rate limited, waiting {wait}s")
                time.sleep(wait)
            elif THINKING_OFF and "thinking" in msg.lower():
                THINKING_OFF = False
            elif use_json and ("response_format" in msg.lower() or "json" in msg.lower()):
                print("  JSON mode rejected; retrying without it")
                use_json = False
            else:
                print("  error:", msg[:150])
                return None
    sys.exit("Still rate limited. Is another script using this key? Progress is saved.")


def data_url(path, max_side=None):
    if max_side:                                   # downscale: fewer image tokens, faster calls
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64," + base64.standard_b64encode(Path(path).read_bytes()).decode()


def label(n, h):
    s = f"[{n}] {h['source']}"
    if h.get("page"):
        s += f" p.{h['page']}"
    if h.get("t_start") is not None:
        s += f" @{h['t_start']:.0f}s"
    return s


def build_content(question, hits, with_images, k_sources=K_SOURCES, max_images=MAX_IMAGES,
                  detail="high", max_side=None):
    content, n_img = [], 0
    for n, h in enumerate(hits[:k_sources], 1):
        if with_images and h.get("image_path") and n_img < max_images:
            content.append({"type": "text", "text": label(n, h)})
            content.append({"type": "image_url",
                            "image_url": {"url": data_url(h["image_path"], max_side), "detail": detail}})
            n_img += 1
        else:
            content.append({"type": "text", "text": f"{label(n, h)}\n{h.get('content') or ''}"})
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


def extract_final(text, json_mode=False):
    t = text.replace("<EOS_TOKEN>", "").strip()
    if json_mode:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                ans = json.loads(m.group(0)).get("answer")
                if isinstance(ans, str) and ans.strip():
                    return ans.strip(), True
            except json.JSONDecodeError:
                pass
    found = re.findall(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if found:
        return found[-1].strip(), True
    return t, False


SYSTEMS = {
    "caption_baseline": dict(retr=lambda db, q: baseline_retrieve(db, q, 20), images=False),
    "fieldnote_routed": dict(retr=lambda db, q: retrieve_routed(db, q, 20), images=True),
    "fieldnote_fast": dict(retr=lambda db, q: retrieve_routed(db, q, 20), images=True,
                           k_sources=8, max_images=2, detail="high", max_side=1000),
    # JSON-mode variants: same retrieval and images, but the output format forbids leaked reasoning
    "caption_baseline_json": dict(retr=lambda db, q: baseline_retrieve(db, q, 20), images=False, json=True),
    "fieldnote_fast_json": dict(retr=lambda db, q: retrieve_routed(db, q, 20), images=True,
                                k_sources=8, max_images=2, detail="high", max_side=1000, json=True),
    "fieldnote_hires_json": dict(retr=lambda db, q: retrieve_routed(db, q, 20), images=True,
                                 k_sources=8, max_images=2, detail="high", max_side=1600, json=True),
}


def load(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save(p, d):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def judge_one(q, a):
    shown = a["final"] if a["tagged"] else "(no answer tags; last part of the output)\n" + a["raw"][-900:]
    prompt = JUDGE_PROMPT.format(q=q["question"], gold=q["gold_answer"], ans=shown)
    for _ in range(3):
        r = chat([{"type": "text", "text": prompt}], max_tokens=400, json_mode=True)
        m = re.search(r"\{.*\}", (r or {}).get("text", ""), re.S)
        if m:
            try:
                j = json.loads(m.group(0))
                if isinstance(j.get("correct"), bool):
                    return j
            except json.JSONDecodeError:
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="caption_baseline,fieldnote_routed")
    ap.add_argument("--retry-bad", action="store_true", help="re-ask answers that were truncated or empty")
    ap.add_argument("--max-tokens", type=int, default=1500)
    args = ap.parse_args()
    sel = [s.strip() for s in args.systems.split(",")]

    qs = [json.loads(l) for l in Path("eval/questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    qs = [q for q in qs if q["type"] != "photo"]
    gold = {q["id"]: q for q in qs}
    answers, judgments = load(ANSWERS), load(JUDGE)

    if args.retry_bad:
        bad = [k for k, a in answers.items()
               if a["system"] in sel and (a["truncated"] or not a["final"].strip())]
        for k in bad:
            answers.pop(k, None)
            judgments.pop(k, None)
        print(f"retrying {len(bad)} truncated/empty answers: {bad}")

    # ---- pass 1: answer ----
    db = DB()
    warmup()
    try:
        for system in sel:
            cfg = SYSTEMS[system]
            use_json = cfg.get("json", False)
            for q in qs:
                key = f"{system}|{q['id']}"
                if key in answers:
                    continue
                t0 = time.perf_counter()
                hits = cfg["retr"](db, q["question"])
                retrieve_ms = (time.perf_counter() - t0) * 1000
                r = chat(build_content(q["question"], hits, cfg["images"], cfg.get("k_sources", K_SOURCES),
                                       cfg.get("max_images", MAX_IMAGES), cfg.get("detail", "high"),
                                       cfg.get("max_side")),
                         system=SYSTEM_PROMPT_JSON if use_json else SYSTEM_PROMPT,
                         max_tokens=args.max_tokens, json_mode=use_json)
                if r is None:
                    print(f"NO ANSWER {key}")
                    continue
                final, tagged = extract_final(r["text"], use_json)
                flags_on = final if tagged else final[-400:]
                answers[key] = {
                    "system": system, "qid": q["id"], "type": q["type"], "raw": r["text"], "final": final,
                    "tagged": tagged, "out_tokens": r["out_tokens"], "truncated": r["truncated"],
                    "retrieve_ms": retrieve_ms, "llm_ms": r["ms"],
                    "cited": bool(re.search(r"\[\d+\]", flags_on)), "abstained": bool(ABSTAIN.search(flags_on)),
                    "top_sources": [label(n, h) for n, h in enumerate(hits[:K_SOURCES], 1)],
                }
                save(ANSWERS, answers)
                print(f"answered {key} ({r['ms']:.0f} ms, {r['out_tokens']} tok, "
                      f"{'ok' if tagged else 'UNPARSED'}{', TRUNCATED' if r['truncated'] else ''})")
    finally:
        db.close()

    # ---- pass 2: judge ----
    for key, a in answers.items():
        if a["system"] not in sel or key in judgments:
            continue
        j = judge_one(gold[a["qid"]], a)
        if j is None:
            print(f"JUDGE FAILED {key}")
            continue
        judgments[key] = j
        save(JUDGE, judgments)
        print(f"judged {key}: {j['correct']}")

    # ---- report ----
    def verdict(s, qid):
        j = judgments.get(f"{s}|{qid}")
        return None if j is None else j["correct"]

    common = [q for q in qs if all(verdict(s, q["id"]) is not None for s in sel)]
    missing = [q["id"] for q in qs if q not in common]
    print(f"\nscored on {len(common)} of {len(qs)} questions that have a verdict for EVERY system"
          + (f"; missing: {missing}" if missing else ""))

    print("\n=== answer accuracy (LLM-judged)")
    print(f"{'group':<8}" + "".join(f"{s:>24}" for s in sel))
    table = {}
    for gname, types in (("text", {"text"}), ("audio", {"audio"}), ("visual", {"visual"}),
                         ("ALL", {"text", "audio", "visual"})):
        ids = [q["id"] for q in common if q["type"] in types]
        row = [sum(verdict(s, i) for i in ids) for s in sel]
        table[gname] = (row, len(ids))
        print(f"{gname:<8}" + "".join(f"{f'{ok}/{len(ids)} ({100 * ok / len(ids):.0f}%)' if ids else '-':>24}"
                                       for ok in row))

    print("\n=== output tokens, truncation, citations, abstentions, latency (retrieve + LLM call, ms)")
    lat = {}
    for s in sel:
        rows = [a for a in answers.values() if a["system"] == s]
        if not rows:
            continue
        toks = [a["out_tokens"] for a in rows if a["out_tokens"] is not None]
        tot = sorted(a["retrieve_ms"] + a["llm_ms"] for a in rows)
        p50, p95 = statistics.median(tot), tot[min(len(tot) - 1, int(0.95 * len(tot)))]
        lat[s] = round(p95)
        print(f"{s:<24} avg out tokens {statistics.mean(toks) if toks else 0:5.0f}   "
              f"truncated {sum(a['truncated'] for a in rows)}   unparsed {sum(not a['tagged'] for a in rows)}   "
              f"cited {100 * sum(a['cited'] for a in rows) / len(rows):3.0f}%   "
              f"abstained {100 * sum(a['abstained'] for a in rows) / len(rows):3.0f}%   "
              f"p50 {p50:6.0f}   p95 {p95:6.0f}")

    if len(sel) >= 2:
        a_sys, b_sys = sel[0], sel[1]
        print(f"\n=== disagreements: {a_sys} vs {b_sys}")
        for q in common:
            va, vb = verdict(a_sys, q["id"]), verdict(b_sys, q["id"])
            if va != vb:
                print(f"\n{q['id']} [{q['type']}] {q['question'][:90]}\n  gold: {q['gold_answer']}")
                for s, v in ((a_sys, va), (b_sys, vb)):
                    ans = answers[f"{s}|{q['id']}"]["final"].replace("\n", " ")[:170]
                    print(f"  {s:<24} {'OK ' if v else 'BAD'} {ans}")

    md = ["| System | n | Visual acc | All acc | avg out tokens | p95 ms (retrieve+LLM) |", "|---|---|---|---|---|---|"]
    for i, s in enumerate(sel):
        rows = [a for a in answers.values() if a["system"] == s]
        toks = [a["out_tokens"] for a in rows if a["out_tokens"] is not None]
        (vr, vn), (ar, an) = table["visual"], table["ALL"]
        md.append(f"| {s} | {an} | {vr[i]}/{vn} | {ar[i]}/{an} | "
                  f"{statistics.mean(toks) if toks else 0:.0f} | {lat.get(s, '-')} |")
    Path("eval/answer_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nwrote eval/answer_results.md")


if __name__ == "__main__":
    main()