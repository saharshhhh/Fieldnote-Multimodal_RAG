# app/answer.py
"""Answer synthesis: numbered sources (text + page/figure images) -> a short cited answer."""
import base64
import io
import mimetypes
import os
import re
import time
from pathlib import Path

from PIL import Image

SYSTEM_PROMPT = (
    "You assist HVAC field technicians. Answer ONLY from the numbered sources. "
    "Read values directly from charts, tables and diagrams when relevant. "
    "Think silently. Output ONLY your final answer inside <answer></answer> tags: one or two sentences, "
    "citing sources like [2]. If the sources don't contain the answer, say so inside the tags."
)

_client = None


def _co():
    """Created lazily so the app runs in retrieval-only mode without an API key."""
    global _client
    if _client is None:
        import cohere

        _client = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    return _client


def model_name():
    return os.environ.get("COHERE_MODEL", "command-a-plus-05-2026")


def data_url(path, max_side=None):
    if max_side:                                   # downscale: fewer image tokens, faster calls
        im = Image.open(path).convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode()
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64," + base64.standard_b64encode(Path(path).read_bytes()).decode()


def label(n, h):
    s = f"[{n}] {h['source']}"
    if h.get("page"):
        s += f" p.{h['page']}"
    if h.get("t_start") is not None:
        s += f" @{h['t_start']:.0f}s"
    return s


def build_content(question, hits, max_images=4, max_side=None):
    """Returns (content blocks, set of source numbers whose IMAGE is attached)."""
    content, seen = [], set()
    for n, h in enumerate(hits, 1):
        path = h.get("image_path")
        if path and len(seen) < max_images and Path(path).exists():
            content.append({"type": "text", "text": label(n, h)})
            content.append({"type": "image_url",
                            "image_url": {"url": data_url(path, max_side), "detail": "high"}})
            seen.add(n)
        else:
            content.append({"type": "text", "text": f"{label(n, h)}\n{h.get('content') or ''}"})
    content.append({"type": "text", "text": f"Question: {question}"})
    return content, seen


def split_answer(raw):
    """Pull the final answer out of the model output.
    Returns (text, followed_format). Falls back to the last paragraph when the model
    wrote its reasoning aloud and skipped the <answer> tags."""
    t = raw.replace("<EOS_TOKEN>", "").strip()
    found = re.findall(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if found and found[-1].strip():
        return found[-1].strip(), True
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    if len(t) > 500 and paras:
        return paras[-1], False
    return t, False


def cited_numbers(text):
    """Source numbers cited in the answer, for [2], [2][3] and [2, 3] styles."""
    nums = set()
    for group in re.findall(r"\[([\d,\s]+)\]", text or ""):
        nums.update(int(x) for x in re.findall(r"\d+", group))
    return nums


def _chat(kwargs):
    co = _co()
    try:
        return co.chat(thinking={"type": "disabled"}, **kwargs)
    except TypeError:                              # older SDK without the `thinking` argument
        return co.chat(**kwargs)
    except Exception as e:
        if "thinking" in str(e).lower():
            return co.chat(**kwargs)
        raise


def answer(question, hits, max_images=4, max_side=None, max_tokens=1500):
    content, seen = build_content(question, hits, max_images, max_side)
    kwargs = dict(
        model=model_name(), max_tokens=max_tokens, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": content}],
    )
    t0 = time.perf_counter()
    resp = _chat(kwargs)
    ms = (time.perf_counter() - t0) * 1000
    raw = "".join(b.text for b in resp.message.content if getattr(b, "type", None) == "text")
    text, tagged = split_answer(raw)
    try:
        out_tokens = int(resp.usage.tokens.output_tokens)
    except Exception:
        out_tokens = None
    fr = str(getattr(resp, "finish_reason", "")).upper()
    return {
        "text": text, "raw": raw, "tagged": tagged, "ms": ms, "out_tokens": out_tokens,
        "seen": seen, "cited": cited_numbers(text), "truncated": "MAX_TOKENS" in fr or "LENGTH" in fr,
    }