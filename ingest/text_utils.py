import re


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph: first by sentences, then by raw characters."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, cur = [], ""
    for s in sentences:
        # a single sentence longer than the limit (tables, run-on spec lines)
        while len(s) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            cut = s.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(s[:cut].strip())
            s = s[cut:].strip()
        if cur and len(cur) + 1 + len(s) > max_chars:
            pieces.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip() if cur else s
    if cur:
        pieces.append(cur)
    return pieces


def split_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """
    Paragraph-aware sliding window.
    - Keeps paragraphs intact when possible.
    - Splits oversized paragraphs by sentence.
    - Carries the last `overlap` chars into the next chunk so facts that straddle
      a boundary still appear whole in at least one chunk.
    A chunk may exceed max_chars by up to `overlap` because of the carried tail.
    """
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        pieces.extend([p] if len(p) <= max_chars else _hard_split(p, max_chars))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + 2 + len(piece) > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap > 0 else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1:]  # start on a word boundary
            cur = f"{tail}\n\n{piece}" if tail else piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks