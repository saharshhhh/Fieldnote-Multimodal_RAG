CREATE EXTENSION IF NOT EXISTS vector;

-- Everything text-embedded: manual text, call transcripts
CREATE TABLE text_chunks (
  id BIGSERIAL PRIMARY KEY,
  modality TEXT NOT NULL CHECK (modality IN ('manual_text','audio')),
  source TEXT NOT NULL,          -- filename
  page INT,                      -- manuals
  t_start REAL, t_end REAL,      -- audio timestamps
  content TEXT NOT NULL,
  embedding VECTOR(384)
);

-- Everything SigLIP-embedded: figure crops, page images, photos
CREATE TABLE image_chunks (
  id BIGSERIAL PRIMARY KEY,
  modality TEXT NOT NULL CHECK (modality IN ('manual_figure','manual_page','photo')),
  source TEXT NOT NULL,
  page INT,
  bbox REAL[],                   -- crop box on the page, for citations
  image_path TEXT NOT NULL,
  context_text TEXT,             -- nearby text from the PDF (not a VLM caption)
  embedding VECTOR(768)
);

CREATE INDEX ON text_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON image_chunks USING hnsw (embedding vector_cosine_ops);