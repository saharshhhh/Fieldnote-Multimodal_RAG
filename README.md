# Fieldnote

**Multimodal RAG for HVAC field technicians.** Ask a question from a rooftop and get an answer
grounded in the *actual page, diagram, or photo* it came from, across product literature,
job-site photos, and manufacturer support calls, each indexed with its own encoder instead of
being flattened into captions.

> A technician asks why a unit keeps tripping. The answer lives in a figure, a label inside a
> drawing, or a recorded support call, not always in a paragraph of text.

This is a portfolio project. It is built end to end (ingestion, retrieval, reranking, answer
synthesis, evaluation), measured against a deliberately strong **caption-only baseline**, and the
write-up includes what did *not* work.

---

## What it demonstrates

- Retrieval across **text, images, and audio** in one system, with a `modality` tag on every chunk.
- **Late fusion** of independent vector spaces (bge text, SigLIP image), with rank fusion and a
  cross-encoder reranker.
- A reproducible **40-question eval** (15 "hard" questions whose evidence is in a figure, table,
  diagram, or photo), a retrieval scorer, an LLM-judged answer eval, and an honest comparison to a
  caption-only pipeline.

## Architecture

```
                 ┌─ bge-small ──────► text_chunks   (manual text, call transcripts)  ─┐
 question ───────┤                                                                    │ top-40 each,
                 └─ SigLIP text ────► image_chunks  (pages, figure crops, photos)    ─┘ in parallel
                                                    │
                                    RRF fusion  →  keep ≤2 candidates per manual page
                                                    │
                                    cross-encoder rerank (candidates that carry text)
                                                    │
                          top-k sources → vision LLM (sees the cited page images)
                                                    │
                                    answer with [n] citations → page / figure images
```

| Piece | Choice |
|---|---|
| PDF parsing | PyMuPDF: text chunks, full-page renders, figure crops (embedded images + clustered vector drawings) |
| Image encoder | SigLIP `google/siglip-base-patch16-224` (768-d), pixels are embedded, not captions |
| Text encoder | `BAAI/bge-small-en-v1.5` (384-d) |
| Audio | `faster-whisper` (small, int8) → timestamped transcripts → bge |
| Vector store | Qdrant **local mode** (no server); one collection per embedding space |
| Fusion | Reciprocal Rank Fusion (rank-based, because scores from different spaces are not comparable) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (CPU) |
| Answer model | Cohere Command A+ (accepts images and text) |

**Design decisions that mattered**

1. **Two embedding spaces, two collections.** A bge vector and a SigLIP vector cannot be compared,
   so each question is encoded twice and searched in parallel.
2. **Rank fusion, not score fusion.** Cosine 0.09 from SigLIP and 0.72 from bge mean different things.
   (The caption baseline *can* merge by score, because everything is embedded with the same model.)
3. **Retrieval finds the page; the vision model reads it.** Embeddings cannot read a label or a table
   value. The synthesis step receives the page images.
4. **A photo-intent router.** Plain RRF interleaves the lists regardless of modality, so a manual
   paragraph took rank 1 even when the user asked to *see a photo* (photo R@1 = 0%). Requests that
   mention a photo/picture search photos only, and weight them 4× over the text list.
5. **Reranker with a candidate-collapse step.** A text cross-encoder cannot read photos, so photo
   requests skip it and figures use their nearby PDF text (not a model-written caption). Figures with
   no nearby text get a neutral score. Candidates from the same page (page image, figure crops, text
   chunks) are collapsed to at most two before reranking.
6. **Speech is transcribed, not "captioned".** Spoken content *is* language, so transcription loses
   nothing; the caption trap applies to modalities where the evidence is not linguistic.

## Corpus

| Modality | Content | Size |
|---|---|---|
| Manuals | Carrier literature: air conditioners, heat pumps, furnaces, thermostats, air purifiers, indoor air quality | 11 PDFs, 112 pages → 311 text chunks, 112 page images, 201 figure crops |
| Photos | HVAC equipment photos from Wikimedia Commons, hand-reviewed | 95 images |
| Audio | Technician-to-support calls, one per product area | 10 calls, **synthetic** |

**Disclosures**

- The support calls are text-to-speech from scripts I wrote (`ingest/make_calls.py`). Every number
  spoken in them is fictional and exists so audio-only evidence can be tested.
- Photos come from Wikimedia Commons under their individual licenses. Attribution for every image is in
  `docs/photo_credits.csv`. The first photo set I downloaded turned out to be pages of an electrical
  code standard; it was discarded.
- The Carrier PDFs are copyrighted and are **not** redistributed here (`data/` is git-ignored).

## Results

All numbers come from a 40-question eval I wrote: **25 text/audio questions and 15 "hard" questions**
(10 about figures, tables, and diagrams; 5 "show me a photo of ..." requests). Gold labels are page-level
(manuals), file-level (calls, photos), and may list several correct pages. Retrieval is scored mechanically
and answer correctness is judged by an LLM against gold answers.

### Retrieval (no LLM calls)

| System | R@1 | R@5 | R@20 | MRR | hard-Q R@20 | hard-Q MRR | latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| Caption-only baseline | 57.5% | 87.5% | 100% | 0.699 | 100% | 0.576 | 26 / 31 ms |
| Fieldnote, plain RRF | 55.0% | 85.0% | 92.5% | 0.664 | 93.3% | 0.624 | 152 / 157 ms |
| + photo router | 62.5% | 90.0% | 92.5% | 0.710 | 93.3% | 0.747 | 153 / 164 ms |
| + reranker, `fast` | 62.5% | 95.0% | 95.0% | 0.726 | 93.3% | 0.702 | 836 / 1013 ms |
| + reranker, `balanced` | 67.5% | 95.0% | **100%** | 0.795 | **100%** | **0.773** | 1837 / 2073 ms |
| + reranker, `high_recall` | **72.5%** | **97.5%** | **100%** | **0.831** | 100% | 0.760 | 4767 / 6557 ms |

Ablations: text space alone reaches R@20 85% overall and 60% on hard questions; the image space alone
reaches 55% overall and 80% on hard questions. The two spaces are complementary (SigLIP is blind to text
lookups, bge is blind to photos).

Latency is retrieval only (CPU-only Windows machine; expect ±15% run to run).

**Takeaways**

- Captioning every image with a vision model and embedding the captions is a **strong retrieval
  baseline** (R@20 = 100%). Retrieval alone does not decide between the two approaches.
- The reranker gives the main retrieval gain. Collapsing near-duplicate candidates before reranking beat
  spending the same compute on more candidates: R@20 100% at ~2 s versus 97.5% at ~2.8 s without collapsing.
- `fast`, `balanced`, and `high_recall` are named presets (`retrieval/rerank.py`) so the latency/recall
  tradeoff is explicit.

### Answer correctness (35 non-photo questions, same model and prompt for both systems)

| System | text | audio | visual | all |
|---|---|---|---|---|
| Caption-only baseline (answers from captions, no images) | 13/15 | 9/10 | 7/10 | 29/35 |
| Fieldnote (answerer sees the cited page images) | 12/15 | 9/10 | **10/10** | **31/35** |

On visual questions Fieldnote answered 3 more correctly and the baseline answered none that Fieldnote
missed (paired sign test p = 0.25). Overall the gap is 4 vs 2 questions (p = 0.69). **This is suggestive,
not statistically significant.** One of the three visual wins (q38) is the baseline failing to finish an
answer, not clearly a wrong one; without it the visual result is 9/9 vs 7/9 (p = 0.5).

Fieldnote's answer run used the RRF + photo-router retrieval, not the reranker. Re-running answers with
the reranked retrieval is pending.

Where captions lost information: the location of a rating plate drawn inside an exploded-view diagram
(the caption did not mention it), and a row of maintenance icons.

### What did not work

- **Diagram labels vs. a text cross-encoder.** The reranker pushed the "(if used)" part-label question
  from rank 3 to rank 11-15. A text model reads a diagram's list of labels as word salad.
- **Downscaling page images to 1000 px** to cut latency lost one small-text question the full-size run
  answered.
- **JSON-mode output** did not stop the model from writing its reasoning into the answer (about half of
  the baseline answers ignored the format). Output length, not retrieval, dominated end-to-end latency.
- **Latency gate not demonstrated.** Retrieval plus reranking stays under ~2 s p95 (`balanced`), but the
  answer step ran from ~1.2 s (short, clean answers) to 8-16 s (when the model reasoned aloud). I could not
  confirm an end-to-end p95 under 6 s with this model on a free Trial key.
- **An early eval run was flawed** (leaked reasoning in answers, truncated judging). It was fixed and rerun;
  the raw outputs of both versions are kept in `eval/`.

### Limitations

- 40 questions written and tuned by one person. The router and reranker settings were chosen after
  seeing results on this same set, and there is no held-out set.
- The brochures contain few true charts or performance curves, so the brief's "chart axis values"
  claim is **not tested** here. The visual questions ask about labels, icons, tables, and drawings.
- The leak check only catches answers that appear verbatim in the PDF text layer. At least one visual
  question (q36) was found by text retrieval after reranking, so its answer may be partly present in the
  text layer.
- Support calls are synthetic; Whisper occasionally mishears words ("thaw" became "fog") and writes
  numbers as digits.
- The answer judge is an LLM; raw verdicts are in `eval/judgments_v2.json`.

## Ship-gate status

| Gate | Status |
|---|---|
| 40-question eval, 15 not answerable from text alone | Done |
| Beat the caption-only baseline on the hard questions by a margin shown in a table | Directional: better hard-Q MRR (0.77 vs 0.58) and visual answers (10/10 vs 7/10); not significant |
| Every answer carries a citation with page number and a cropped image | Done in the demo app: each `[n]` maps to a source card with the page number and the figure crop, full page, photo, or call timestamp |
| p95 < 6 s at k=20 | Retrieval yes; end-to-end not demonstrated |

## Quickstart (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

$env:COHERE_API_KEY = "your_key"        # only needed for captions, drafts, answers, judging
# optional: $env:HF_TOKEN = "..."       # silences Hugging Face download warnings
```

Put your PDFs in `data/manuals/`, then:

```powershell
python -m ingest.make_calls              # synthetic support calls + data/calls/gold_facts.json
python -m ingest.download_photos         # Wikimedia Commons photos + manifest.csv (set a real contact email in the script)
python -m ingest.review_photos           # contact sheets: delete mismatches with ingest.drop_photos
python -m ingest.run_all --reset         # builds the Qdrant index in data/qdrant
python -m retrieval.try_search "why does the indoor coil keep freezing"
```

`run_all` transcribes calls first and then loads the image/text encoders, which keeps peak memory
down on small machines. Run `python -m ingest.run_all --only photos` (or `manuals`, `calls`) to add one
modality without `--reset`; there is no de-duplication.

## Demo app

```powershell
streamlit run app/main.py
```

Ask a question and get a short answer with numbered citations. Every `[n]` maps to a source card showing
the evidence itself: the figure crop or full manual page, the photo (with its license and author), or the
support call with an audio player that starts at the cited timestamp. Cards are tagged when they were cited
in the answer and when the model actually saw their image.

- A sidebar preset menu switches between the `fast`, `balanced`, and `high_recall` rerankers and the
  no-reranker path, so the latency/recall tradeoff can be felt directly.
- Photo requests ("show me a photo of ...") skip the language model and return a photo grid.
- Without `COHERE_API_KEY` the app runs in retrieval-only mode, and a model-API failure never hides the
  retrieved sources.
- Local Qdrant allows one open client, so close eval scripts while the app is running.

## Reproducing the evaluation

```powershell
# questions
python -m eval.make_auto_questions       # audio + photo questions (no API)
python -m eval.draft_questions           # model-drafted text/visual questions (API); verify every draft by hand
python -m eval.finalize                  # merge, run the text-leak check, write eval/questions.jsonl

# caption-only baseline (about 410 API calls)
python -m eval.caption_all
python -m eval.build_baseline

# scoring
python -m eval.score                     # retrieval metrics + latency, no API calls; writes eval/results.md
python -m eval.rerank_sweep              # reranker speed/recall tradeoff, no API calls
python -m eval.answer_eval               # answer accuracy with an LLM judge (~140 API calls per two systems)
python -m eval.compare                   # paired sign test on the answer verdicts
```

> **API budget:** a Cohere Trial key allows 20 calls/minute and 1,000 calls/month. Captioning plus two
> answer-and-judge passes uses most of a month. The scripts throttle, save progress after every call, and
> resume where they stopped.

## Repository layout

```
ingest/       PDF/audio/photo ingestion, encoders, Qdrant wrapper, photo curation tools
retrieval/    parallel search, RRF fusion, photo router, cross-encoder reranker + presets
eval/         question building, caption baseline, retrieval scorer, answer eval, comparison scripts
app/          Streamlit demo (main.py), answer synthesis (answer.py), shared index handle (runtime.py)
docs/         photo_credits.csv
data/         (git-ignored) manuals, photos, calls, rendered pages, Qdrant index
```

## Roadmap

- Re-run the answer eval with reranked retrieval and a production API key; measure end-to-end p95.
- A held-out question set, and a corpus that includes real charts and performance curves.
- Caption-as-extra-retrieval-channel ablation (captions found one question that pixels alone missed).
- A relevance threshold so the system abstains instead of returning a nearest neighbour when nothing is relevant.
- A quantized/ONNX cross-encoder for lower reranking latency.

## Licensing

Code: MIT (add a `LICENSE` file). Photos: individual Wikimedia Commons licenses, see
`docs/photo_credits.csv`. Carrier manuals are copyrighted by their owner and are not included.