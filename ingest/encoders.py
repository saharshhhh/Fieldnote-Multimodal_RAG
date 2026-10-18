# ingest/encoders.py
# pip install torch transformers sentence-transformers pillow sentencepiece protobuf
from functools import lru_cache

import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SIGLIP_ID = "google/siglip-base-patch16-224"   # 768-dim -> image_chunks.embedding VECTOR(768)
TEXT_ID = "BAAI/bge-small-en-v1.5"             # 384-dim -> text_chunks.embedding  VECTOR(384)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------- lazy model loading (loads once, reused for every request) ----------

@lru_cache(maxsize=1)
def _siglip():
    model = AutoModel.from_pretrained(SIGLIP_ID).to(DEVICE).eval()
    processor = AutoProcessor.from_pretrained(SIGLIP_ID)
    return model, processor


@lru_cache(maxsize=1)
def _text_model():
    return SentenceTransformer(TEXT_ID, device=DEVICE)


def warmup():
    """Call once at app startup so the first user request doesn't pay model-load time."""
    embed_image_query("warmup")
    embed_text_query("warmup")


def _as_tensor(out):
    """Different transformers versions return a tensor or a model-output object."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output


# ---------- SigLIP: the image space ----------

@torch.no_grad()
def embed_images(paths: list[str]) -> list[list[float]]:
    """Used at INGEST time: image files -> 768-dim unit vectors."""
    model, processor = _siglip()
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    feats = _as_tensor(model.get_image_features(**inputs))
    return F.normalize(feats, dim=-1).cpu().tolist()


@torch.no_grad()
def embed_image_query(text: str) -> list[float]:
    """Used at QUERY time: a text question -> 768-dim vector in SigLIP's image space.
    This uses SigLIP's TEXT tower, which was trained to land near matching images."""
    model, processor = _siglip()
    inputs = processor(
        text=[text],
        padding="max_length",   # SigLIP was trained with fixed-length padding; do not drop this
        max_length=64,          # SigLIP's text tower reads at most 64 tokens
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    feats = _as_tensor(model.get_text_features(**inputs))
    return F.normalize(feats, dim=-1)[0].cpu().tolist()


# ---------- bge: the text space ----------

def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Used at INGEST time: manual text chunks and call transcripts -> 384-dim unit vectors.
    Documents get NO prefix."""
    return _text_model().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def embed_text_query(q: str) -> list[float]:
    """Used at QUERY time: a question -> 384-dim vector. bge wants the instruction prefix
    on queries only."""
    return _text_model().encode(
        BGE_QUERY_PREFIX + q, normalize_embeddings=True
    ).tolist()