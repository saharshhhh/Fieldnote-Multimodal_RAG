import sys
# pyrefly: ignore [missing-import]
import pymupdf as fitz
from ingest.figures import find_figures

pdf_path, page_no = sys.argv[1], int(sys.argv[2])  # 1-indexed page
doc = fitz.open(pdf_path)
page = doc[page_no - 1]

colors = {"raster": (0, 0.6, 0), "vector": (0.9, 0, 0), "mixed": (0, 0.2, 0.9)}
for bbox, kind in find_figures(page):
    page.draw_rect(fitz.Rect(bbox), color=colors[kind], width=2)
    page.insert_text((bbox[0] + 4, bbox[1] + 12), kind, color=colors[kind], fontsize=9)

out = f"debug_p{page_no}.png"
page.get_pixmap(dpi=110).save(out)
print("wrote", out)