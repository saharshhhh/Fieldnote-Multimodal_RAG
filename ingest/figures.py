# pyrefly: ignore [missing-import]
import pymupdf as fitz  # PyMuPDF


def _expand(r: fitz.Rect, d: float) -> fitz.Rect:
    return fitz.Rect(r.x0 - d, r.y0 - d, r.x1 + d, r.y1 + d)


def _merge_boxes(items: list[tuple[fitz.Rect, set]], gap: float):
    """Repeatedly merge boxes that overlap or sit within `gap` points of each other.
    Needed because one chart is often many pieces: axes, curve, legend, raster logo."""
    changed = True
    while changed:
        changed = False
        out: list[tuple[fitz.Rect, set]] = []
        for rect, kinds in items:
            for i, (orect, okinds) in enumerate(out):
                if _expand(rect, gap).intersects(orect):
                    out[i] = (orect | rect, okinds | kinds)  # union of rectangles
                    changed = True
                    break
            else:
                out.append((fitz.Rect(rect), set(kinds)))
        items = out
    return items


def find_figures(
    page: fitz.Page,
    min_size: float = 60,         # points; ignore anything smaller than this in w or h
    min_area_frac: float = 0.015, # ignore boxes covering <1.5% of the page
    max_area_frac: float = 0.92,  # ignore page-sized backgrounds/borders
    max_aspect: float = 10.0,     # ignore thin rules and separators
    merge_gap: float = 12,
    pad: float = 6,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Return [(bbox, kind), ...] where kind is 'raster', 'vector' or 'mixed'."""
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    candidates: list[tuple[fitz.Rect, set]] = []

    # 1) embedded raster images (photos, scanned diagrams)
    for info in page.get_image_info():
        candidates.append((fitz.Rect(info["bbox"]), {"raster"}))

    # 2) vector graphics (most wiring diagrams and pressure curves are drawn this way)
    try:
        for r in page.cluster_drawings():
            candidates.append((fitz.Rect(r), {"vector"}))
    except Exception:
        pass  # very old PyMuPDF without cluster_drawings; raster-only fallback

    merged = _merge_boxes(candidates, merge_gap)

    figures = []
    for rect, kinds in merged:
        rect = rect & page_rect  # clip to page
        if rect.is_empty:
            continue
        w, h = rect.width, rect.height
        area_frac = (w * h) / page_area
        if w < min_size or h < min_size:
            continue
        if not (min_area_frac <= area_frac <= max_area_frac):
            continue
        if max(w / h, h / w) > max_aspect:
            continue
        padded = _expand(rect, pad) & page_rect
        kind = "mixed" if len(kinds) > 1 else next(iter(kinds))
        figures.append(((padded.x0, padded.y0, padded.x1, padded.y1), kind))

    figures.sort(key=lambda f: (f[0][1], f[0][0]))  # reading order: top to bottom, left to right
    return figures


def text_near(page: fitz.Page, bbox, margin: float = 50, max_chars: int = 600) -> str:
    """Text inside and just above/below a figure: captions, axis labels, 'Figure 12: ...'.
    This is original PDF text, not a model-generated caption."""
    box = fitz.Rect(bbox)
    zone = fitz.Rect(box.x0 - 10, box.y0 - margin, box.x1 + 10, box.y1 + margin)
    parts = []
    for x0, y0, x1, y1, txt, _no, btype in page.get_text("blocks", sort=True):
        if btype != 0:  # 0 = text block, 1 = image block
            continue
        if fitz.Rect(x0, y0, x1, y1).intersects(zone):
            t = " ".join(txt.split())
            if t:
                parts.append(t)
    return " ".join(parts)[:max_chars]