# ingest/manuals.py
from pathlib import Path

import pymupdf as fitz

from ingest.figures import find_figures, text_near
from ingest.text_utils import split_text


def ingest_manual(pdf_path, out_dir, db):
    pdf_path, out = Path(pdf_path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem
    doc = fitz.open(str(pdf_path))
    n_text = n_pages = n_figs = 0

    for pno, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()

        # 1) text chunks
        for chunk in split_text(text, max_chars=1200, overlap=150):
            db.insert_text("manual_text", pdf_path.name, pno, chunk)
            n_text += 1

        # 2) full page image (retrieval fallback AND the citation image shown to the tech)
        page_png = out / f"{stem}_p{pno}.png"
        page.get_pixmap(dpi=170).save(str(page_png))
        db.insert_image("manual_page", pdf_path.name, pno, page_png.as_posix(),
                        bbox=None, context=text[:600])
        n_pages += 1

        # 3) figure crops (charts, diagrams, spec-table graphics)
        for i, (bbox, _kind) in enumerate(find_figures(page)):
            crop_png = out / f"{stem}_p{pno}_f{i}.png"
            page.get_pixmap(dpi=200, clip=fitz.Rect(*bbox)).save(str(crop_png))
            db.insert_image("manual_figure", pdf_path.name, pno, crop_png.as_posix(),
                            bbox=list(bbox), context=text_near(page, bbox))
            n_figs += 1

    doc.close()
    print(f"{pdf_path.name}: {n_text} text chunks, {n_pages} pages, {n_figs} figure crops")