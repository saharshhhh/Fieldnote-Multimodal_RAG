# app/answer.py
import base64
import mimetypes
import os

import cohere

co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])

# Model ID reported by third-party listings; confirm it in your Cohere dashboard / models page.
MODEL = os.environ.get("COHERE_MODEL", "command-a-plus-05-2026")

SYSTEM_PROMPT = (
    "You assist HVAC field technicians. Answer ONLY from the numbered sources. "
    "Read values directly from charts/diagrams when relevant. Cite as [n]. "
    "If the sources don't contain the answer, say so."
)


def image_to_data_url(path: str) -> str:
    """Build a data URL with the CORRECT mime type (photos are .jpg, rendered pages are .png)."""
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def answer(question: str, hits: list[dict], max_images: int = 4, max_sources: int = 8):
    content: list[dict] = []
    sources: list[dict] = []
    n_images = 0

    for n, h in enumerate(hits[:max_sources], start=1):
        label = f"[{n}] {h['source']}" + (f" p.{h['page']}" if h.get("page") else "")
        if h.get("t_start") is not None:
            label += f" @{h['t_start']:.0f}s"  # audio citations point at a timestamp

        if h.get("image_path") and n_images < max_images:
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {"url": image_to_data_url(h["image_path"]), "detail": "high"},
            })
            n_images += 1
        else:
            content.append({"type": "text", "text": f"{label}\n{h.get('content') or ''}"})
        sources.append(h)

    content.append({"type": "text", "text": f"Question: {question}"})

    resp = co.chat(
        model=MODEL,
        max_tokens=1000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )

    # v2 responses are a list of content blocks (may include reasoning blocks); keep only text
    text = "".join(
        block.text for block in resp.message.content if getattr(block, "type", None) == "text"
    )
    return text.strip(), sources