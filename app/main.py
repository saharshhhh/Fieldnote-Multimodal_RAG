# app/main.py
# Run from the project root:  streamlit run app/main.py
import csv
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # index stores relative paths like data/pages/...

import streamlit as st  # noqa: E402

from app.answer import answer as generate_answer  # noqa: E402
from app.runtime import boot, get_db  # noqa: E402

st.set_page_config(page_title="Fieldnote", page_icon="🔧", layout="wide")

PRESETS = {
    "balanced: reranked, ~2 s (recommended)": "balanced",
    "fast: reranked, ~1 s": "fast",
    "high recall: reranked, ~5 s": "high_recall",
    "no reranker: RRF + photo router, ~0.15 s": "routed",
}
# Relevance cutoff on the top reranker score. Below it the app refuses instead of calling the model.
# Calibrate with `python -m eval.calibrate_threshold`, then set FIELDNOTE_MIN_RERANK (-12 = off).
DEFAULT_MIN_RERANK = float(os.environ.get("FIELDNOTE_MIN_RERANK", "-12"))
IMAGE_SIZES = {"original": None, "1600 px": 1600, "1000 px (faster, may blur small labels)": 1000}
BADGE = {
    "manual_text": "📄 manual text", "audio": "🎧 support call", "manual_page": "🗒️ manual page",
    "manual_figure": "📊 figure", "photo": "📷 photo",
}
EXAMPLES = [
    "Where is the rating plate located on the OM9GFRC furnace?",
    "What should be done with the remote control batteries before long non-use of the 45MBAAQ?",
    "How long should a frozen indoor coil thaw with the compressor off?",
    "Show me a photo of a motor run capacitor",
    "What filter size does the OM9GFRC use with a 20 inch cabinet height?",
]


@st.cache_resource(show_spinner="Loading the index and models (first start only)...")
def _boot():
    boot()
    return True


@st.cache_data(show_spinner=False)
def index_counts():
    return get_db().counts()


@st.cache_data(show_spinner=False)
def photo_credits():
    path = Path("data/photos/manifest.csv")
    out = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["file"]] = row
    return out


def photo_intent(q):
    from retrieval.search import PHOTO_INTENT
    return bool(PHOTO_INTENT.search(q))


def run_retrieval(q, mode, k=20):
    from retrieval.rerank import retrieve_preset
    from retrieval.search import retrieve_routed

    db = get_db()
    if mode == "routed":
        return retrieve_routed(db, q, k)
    return retrieve_preset(db, q, k, mode)


def page_image_path(source, page):
    if not page:
        return None
    p = Path("data/pages") / f"{Path(source).stem}_p{page}.png"
    return p if p.exists() else None


def score_text(h):
    if "rerank" in h:
        return f"rerank score {h['rerank']:.2f}"
    if "rrf" in h:
        return f"fusion score {h['rrf']:.4f}"
    return f"similarity {h.get('score', 0):.3f}"


def where_text(h):
    if h.get("page"):
        return f"page {h['page']}"
    if h.get("t_start") is not None:
        return f"{h['t_start']:.0f} s into the call"
    return ""


def describe_error(e):
    msg = str(e)
    if "429" in msg:
        return ("The model API refused the request (rate limit or monthly cap on the key). "
                "The retrieved sources below are still valid.")
    if isinstance(e, KeyError) and "COHERE_API_KEY" in msg:
        return "COHERE_API_KEY is not set."
    return f"The answer step failed: {msg[:300]}"


def render_source(n, h, cited, seen):
    credits = photo_credits()
    with st.container(border=True):
        tags = [f"**[{n}] {BADGE.get(h['modality'], h['modality'])}**", f"`{h['source']}`", where_text(h)]
        if cited:
            tags.append("✅ cited in the answer")
        if seen:
            tags.append("👁 image seen by the model")
        st.markdown(" · ".join(t for t in tags if t))
        st.caption(score_text(h))

        mod = h["modality"]
        if mod == "audio":
            audio_path = Path("data/calls") / h["source"]
            if audio_path.exists():
                try:
                    st.audio(str(audio_path), start_time=int(h.get("t_start") or 0))
                except TypeError:
                    st.audio(str(audio_path))
            st.write(h.get("content") or "")
        elif mod == "manual_text":
            st.write((h.get("content") or "")[:700])
            page_png = page_image_path(h["source"], h.get("page"))
            if page_png:
                with st.expander("Show the source page"):
                    st.image(str(page_png), width=640)
        elif mod in ("manual_figure", "manual_page"):
            if h.get("image_path") and Path(h["image_path"]).exists():
                st.image(h["image_path"], width=640,
                         caption="figure crop" if mod == "manual_figure" else "full page")
            if mod == "manual_figure":
                page_png = page_image_path(h["source"], h.get("page"))
                if page_png:
                    with st.expander("Show the full page"):
                        st.image(str(page_png), width=640)
        elif mod == "photo":
            if h.get("image_path") and Path(h["image_path"]).exists():
                st.image(h["image_path"], width=420)
            c = credits.get(h["source"])
            if c:
                st.caption(f"Photo: {c.get('artist') or 'unknown'} · {c.get('license') or 'license n/a'} · "
                           f"[source]({c.get('source')})")


def ask(q, mode, n_sources, max_images, max_side, want_answer, min_rerank):
    res = {"q": q, "mode": mode, "n_sources": n_sources, "photo": photo_intent(q),
           "answer": None, "error": None, "low_relevance": False, "top_rerank": None,
           "min_rerank": min_rerank}
    with st.spinner("Searching the manuals, photos and calls..."):
        t0 = time.perf_counter()
        res["hits"] = run_retrieval(q, mode)
        res["ret_ms"] = (time.perf_counter() - t0) * 1000

    top = res["hits"][0].get("rerank") if res["hits"] else None
    res["top_rerank"] = top
    res["low_relevance"] = (not res["photo"]) and top is not None and top < min_rerank

    if want_answer and not res["photo"] and not res["low_relevance"]:
        if not os.environ.get("COHERE_API_KEY"):
            res["error"] = "COHERE_API_KEY is not set, so only retrieval ran."
        else:
            with st.spinner("Reading the sources with the vision model..."):
                try:
                    res["answer"] = generate_answer(q, res["hits"][:n_sources], max_images, max_side)
                except Exception as e:  # noqa: BLE001
                    res["error"] = describe_error(e)
    st.session_state["result"] = res


# ---------------------------------------------------------------- sidebar
_boot()
with st.sidebar:
    st.title("🔧 Fieldnote")
    st.caption("Multimodal RAG for HVAC field technicians")
    preset_label = st.selectbox("Retrieval preset", list(PRESETS))
    have_key = bool(os.environ.get("COHERE_API_KEY"))
    want_answer = st.checkbox("Generate an answer (calls the vision model)", value=have_key,
                              disabled=not have_key,
                              help="Needs COHERE_API_KEY. Off = retrieval only, no API calls.")
    if not have_key:
        st.info("No COHERE_API_KEY: retrieval-only mode.")
    n_sources = st.slider("Sources to show / send to the model", 3, 12, 8)
    max_images = st.slider("Images sent to the model", 0, 4, 4)
    size_label = st.selectbox("Image size sent to the model", list(IMAGE_SIZES))
    min_rerank = st.slider(
        "Relevance cutoff (top reranker score)", -12.0, 8.0, DEFAULT_MIN_RERANK, 0.5,
        help="If the best source scores below this, the app says nothing relevant was found and does "
             "not call the model. -12 = off. Calibrate with eval/calibrate_threshold.py. "
             "Only applies to the reranked presets.")
    with st.expander("Index"):
        for r in index_counts():
            st.write(f"{r['modality']}: {r['n']}")
    st.caption("Close other scripts that use data/qdrant while this app is running; "
               "local mode allows one open client.")

# ---------------------------------------------------------------- main
st.header("Ask the manuals, photos and support calls")

trigger = None
cols = st.columns(len(EXAMPLES))
for i, (c, ex) in enumerate(zip(cols, EXAMPLES)):
    if c.button(f"Example {i + 1}", help=ex, key=f"ex{i}"):
        trigger = ex
        st.session_state["q_text"] = ex

with st.form("ask_form"):
    q_in = st.text_input("Question", value=st.session_state.get("q_text", ""),
                         placeholder="e.g. What does error code EC07 mean?")
    submitted = st.form_submit_button("Ask")

question = trigger or (q_in.strip() if submitted else None)
if question:
    ask(question, PRESETS[preset_label], n_sources, max_images, IMAGE_SIZES[size_label], want_answer, min_rerank)

res = st.session_state.get("result")
if not res:
    st.info("Type a question or click an example. Answers cite numbered sources; each source shows "
            "the exact page, figure crop, photo, or call timestamp behind it.")
    st.stop()

st.markdown(f"**Question:** {res['q']}")
timing = [f"retrieval {res['ret_ms']:.0f} ms"]
if res["answer"]:
    timing.append(f"answer {res['answer']['ms']:.0f} ms")
    if res["answer"]["out_tokens"] is not None:
        timing.append(f"{res['answer']['out_tokens']} output tokens")
st.caption(" · ".join(timing) + f" · preset: {res['mode']}")

hits = res["hits"]
n = res["n_sources"]

if res["photo"]:
    st.subheader("Matching photos")
    st.caption("Photo request: the photo router searched photos only, so no answer is generated.")
    credits = photo_credits()
    photos = [h for h in hits if h["modality"] == "photo"][:8]
    for row_start in range(0, len(photos), 4):
        for col, h in zip(st.columns(4), photos[row_start:row_start + 4]):
            with col:
                st.image(h["image_path"], width=220)
                c = credits.get(h["source"])
                st.caption(f"{h['source']}" + (f" · {c['license']}" if c and c.get("license") else ""))
    with st.expander("Photo credits"):
        for h in photos:
            c = credits.get(h["source"])
            if c:
                st.write(f"{h['source']}: {c.get('artist') or 'unknown'}, {c.get('license') or ''}, "
                         f"[source]({c.get('source')})")
    st.stop()

if res["low_relevance"]:
    st.warning(
        f"No relevant source found. The best match scored {res['top_rerank']:.2f}, below the cutoff of "
        f"{res['min_rerank']:.1f}, so the model was not called. Try rephrasing, or lower the cutoff in the "
        "sidebar if this question should be answerable.")
    st.subheader("Closest results (low relevance, not sent to the model)")
    for i, h in enumerate(hits[:3], 1):
        render_source(i, h, False, False)
    st.stop()

cited, seen = set(), set()
if res["answer"]:
    a = res["answer"]
    cited, seen = a["cited"], a["seen"]
    st.subheader("Answer")
    st.markdown(a["text"])
    if not a["tagged"]:
        st.warning("The model ignored the answer format and wrote its reasoning aloud; "
                   "showing the end of its output.")
    if a["truncated"]:
        st.warning("The model hit the output limit before finishing.")
    with st.expander("Raw model output"):
        st.code(a["raw"])
elif res["error"]:
    st.warning(res["error"])
elif not want_answer:
    st.caption("Retrieval-only mode: tick 'Generate an answer' in the sidebar to add a cited answer.")

st.subheader(f"Sources (top {min(n, len(hits))})")
for i, h in enumerate(hits[:n], 1):
    render_source(i, h, i in cited, i in seen)

rest = hits[n:]
if rest:
    with st.expander(f"More results ({len(rest)})"):
        for i, h in enumerate(rest, n + 1):
            st.write(f"[{i}] {BADGE.get(h['modality'], h['modality'])} · {h['source']} · "
                     f"{where_text(h)} · {score_text(h)}")