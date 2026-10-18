# ingest/db.py
import os
import uuid

from qdrant_client import QdrantClient, models

PATH = os.environ.get("QDRANT_PATH", "data/qdrant")

TEXT_COLLECTION = "text_chunks"     # bge-small vectors, 384-d
IMAGE_COLLECTION = "image_chunks"   # SigLIP vectors, 768-d

TEXT_MODALITIES = ("manual_text", "audio")
IMAGE_MODALITIES = ("manual_figure", "manual_page", "photo")


class DB:
    def __init__(self, path: str | None = None):
        self.client = QdrantClient(path=path or PATH)   # local mode: just a folder on disk
        self._ensure()

    # ---- collections ----

    def _ensure(self):
        existing = {c.name for c in self.client.get_collections().collections}
        for name, dim in ((TEXT_COLLECTION, 384), (IMAGE_COLLECTION, 768)):
            if name not in existing:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                )

    def reset(self):
        for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
            if self.client.collection_exists(name):
                self.client.delete_collection(name)
        self._ensure()

    def close(self):
        self.client.close()

    # ---- writes (embedding happens here so ingest code stays simple) ----

    def insert_text(self, modality, source, page, content,
                    t_start=None, t_end=None, vec=None):
        from ingest.encoders import embed_texts   # lazy import: DB() stays fast to open
        vec = vec or embed_texts([content])[0]
        self.client.upsert(
            collection_name=TEXT_COLLECTION,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "modality": modality, "source": source, "page": page,
                    "t_start": t_start, "t_end": t_end, "content": content,
                },
            )],
        )

    def insert_image(self, modality, source, page, image_path,
                     bbox=None, context=None, vec=None):
        from ingest.encoders import embed_images
        vec = vec or embed_images([image_path])[0]
        self.client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=[models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "modality": modality, "source": source, "page": page,
                    "bbox": bbox, "image_path": image_path, "context_text": context,
                },
            )],
        )

    # ---- reads ----

    @staticmethod
    def _modality_filter(modalities):
        if not modalities:
            return None
        return models.Filter(must=[models.FieldCondition(
            key="modality", match=models.MatchAny(any=list(modalities)))])

    def search_text(self, vec, k=20, modalities=None):
        res = self.client.query_points(
            collection_name=TEXT_COLLECTION, query=vec, limit=k,
            query_filter=self._modality_filter(modalities), with_payload=True,
        )
        return [{
            "id": p.id, "modality": p.payload["modality"], "source": p.payload["source"],
            "page": p.payload.get("page"), "t_start": p.payload.get("t_start"),
            "content": p.payload["content"], "image_path": None, "score": p.score,
        } for p in res.points]

    def search_images(self, vec, k=20, modalities=None):
        res = self.client.query_points(
            collection_name=IMAGE_COLLECTION, query=vec, limit=k,
            query_filter=self._modality_filter(modalities), with_payload=True,
        )
        return [{
            "id": p.id, "modality": p.payload["modality"], "source": p.payload["source"],
            "page": p.payload.get("page"), "t_start": None,
            "content": p.payload.get("context_text"), "image_path": p.payload["image_path"],
            "score": p.score,
        } for p in res.points]

    def counts(self) -> list[dict]:
        rows = []
        for coll, mods in ((TEXT_COLLECTION, TEXT_MODALITIES), (IMAGE_COLLECTION, IMAGE_MODALITIES)):
            for m in mods:
                n = self.client.count(
                    collection_name=coll, count_filter=self._modality_filter([m]), exact=True,
                ).count
                rows.append({"modality": m, "n": n})
        return sorted(rows, key=lambda r: r["modality"])