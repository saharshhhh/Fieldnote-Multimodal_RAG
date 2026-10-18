# ingest/download_photos.py
import csv
import re
import time
from collections import Counter
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "FieldnoteStudentProject/0.1 (personal portfolio; contact: pabbathisaharsh@gmail.com)"}
OUT = Path("data/photos")
OUT.mkdir(parents=True, exist_ok=True)

# search term -> how many photos to keep (the term is only a WEAK label; verify by eye)
TERMS = {
    "furnace flame sensor": 8,
    "gas furnace burner": 12,
    "furnace heat exchanger": 8,
    "furnace inducer motor": 6,
    "pressure switch furnace": 6,
    "heat pump outdoor unit": 14,
    "air conditioner condenser unit": 14,
    "condenser coil": 8,
    "evaporator coil": 10,
    "frozen evaporator coil": 6,
    "ice on heat pump": 6,
    "run capacitor": 10,
    "burnt contactor": 6,
    "contactor relay": 8,
    "thermostat wiring": 10,
    "programmable thermostat": 8,
    "air filter dirty": 10,
    "air purifier": 10,
    "ultraviolet air purifier": 4,
    "humidifier": 6,
    "energy recovery ventilator": 6,
    "rooftop unit hvac": 10,
    "refrigerant manifold gauges": 8,
    "condensate drain line": 6,
    "refrigerant line set": 6,
    "ductwork hvac": 8,
}

OK_MIME = {"image/jpeg": ".jpg", "image/png": ".png"}
BAD_TITLE = re.compile(r"\b(diagram|logo|icon|map|page|scan|svg|chart|pdf|drawing|schematic)\b", re.I)
PAGE_SIZE = 50
MAX_OFFSET = 150   # look at up to 200 results per term


def strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def api_search(query: str, offset: int):
    params = {
        "action": "query", "format": "json", "formatversion": 2,
        "generator": "search", "gsrnamespace": 6, "gsrsearch": query,
        "gsrlimit": PAGE_SIZE, "gsroffset": offset,
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 900,
    }
    r = requests.get(API, params=params, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code}: {r.text[:150]}")
        return []
    data = r.json()
    if "error" in data:
        print("  API error:", data["error"])
        return []
    pages = data.get("query", {}).get("pages", [])
    return sorted(pages, key=lambda p: p.get("index", 0))


def candidates(term: str):
    """Yield candidate files: image-only search first, plain search as a fallback."""
    got_any = False
    for query in (f"{term} filetype:bitmap", term):
        for offset in range(0, MAX_OFFSET + 1, PAGE_SIZE):
            batch = api_search(query, offset)
            if not batch:
                break
            got_any = True
            yield from batch
        if got_any:
            return   # the filtered query worked; don't fall back


def main():
    for old in OUT.glob("photo_*"):   # clear leftovers from earlier partial runs
        old.unlink()

    manifest, seen, total = [], set(), 0
    for term, want in TERMS.items():
        why, kept, examined = Counter(), 0, 0
        for page in candidates(term):
            if kept >= want:
                break
            examined += 1
            info = (page.get("imageinfo") or [None])[0]
            title = page.get("title", "")
            if not info:
                why["no imageinfo"] += 1
                continue
            mime = info.get("thumbmime") or info.get("mime")
            url = info.get("thumburl") or info.get("url")
            if mime not in OK_MIME or not url:
                why[f"mime {mime}"] += 1
                continue
            if BAD_TITLE.search(title):
                why["diagram/page-like title"] += 1
                continue
            if info.get("width", 0) < 500 or info.get("height", 0) < 400:
                why["too small"] += 1
                continue
            if title in seen:
                why["duplicate"] += 1
                continue

            fname = f"photo_{total + 1:03d}{OK_MIME[mime]}"
            try:
                img = requests.get(url, headers=HEADERS, timeout=60)
                img.raise_for_status()
            except Exception as e:
                why["download failed"] += 1
                print(f"  download failed for {title}: {e}")
                continue
            (OUT / fname).write_bytes(img.content)

            meta = info.get("extmetadata", {})
            manifest.append({
                "file": fname, "search_term": term, "commons_title": title,
                "license": strip_html(meta.get("LicenseShortName", {}).get("value", "")),
                "artist": strip_html(meta.get("Artist", {}).get("value", "")),
                "source": info.get("descriptionurl", ""),
            })
            seen.add(title)
            kept += 1
            total += 1
            time.sleep(0.4)   # be polite to Wikimedia

        print(f"{term!r}: examined {examined}, kept {kept}/{want}"
              + (f"  | rejected: {dict(why)}" if why else ""))
        time.sleep(0.3)

    if not manifest:
        print("\nNothing downloaded. Paste the lines above and I'll diagnose it.")
        return
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\n{total} photos saved to {OUT}/  (attribution in manifest.csv)")


if __name__ == "__main__":
    main()