"""File contract shared by snapshot preparation and deployment."""

from __future__ import annotations

MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)

RETRIEVAL_FILES = (
    "answer_ids.npy",
    "corpus.parquet",
    "embeddings_fp16.npy",
    "manifest.json",
)

REQUIRED_FILES = MODEL_FILES + tuple(f"demo/{name}" for name in RETRIEVAL_FILES)
