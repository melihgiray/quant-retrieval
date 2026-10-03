"""Evaluate saved document vectors while encoding only incoming queries."""

from pathlib import Path

import numpy as np
import pandas as pd

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.retrieval.checkpoint import verify_checkpoint
from quant_retrieval.retrieval.dense import DenseRetriever
from quant_retrieval.retrieval.index_artifacts import corpus_ids, read_manifest, verify_export_files


class PrecomputedDenseRetriever(DenseRetriever):
    def __init__(self, model_name: str, artifacts_path: str | Path, precision: str = "fp16",
                 device: str = "auto", batch_size: int = 64, show_progress: bool = False):
        if precision not in {"fp16", "fp32"}:
            raise ValueError("precision must be fp16 or fp32")
        self.artifacts_path = Path(artifacts_path)
        path = self.artifacts_path / "manifest.json"
        self.manifest = read_manifest(path)
        for key in ("documents", "dimensions", "max_length"):
            if type(self.manifest.get(key)) is not int or self.manifest[key] <= 0:
                raise ValueError(f"artifact {key} must be a positive integer")
        checkpoint = self.manifest.get("checkpoint")
        self.checkpoint_verified = verify_checkpoint(Path(model_name), self.manifest)
        if not self.checkpoint_verified and (not isinstance(checkpoint, str)
                or Path(checkpoint).resolve() != Path(model_name).resolve()):
            raise ValueError("query model must match the exported checkpoint path")
        self.precision = precision
        super().__init__(model_name, batch_size=batch_size, device=device,
                         max_length=self.manifest["max_length"],
                         pooling=self.manifest.get("pooling", "mean"), show_progress=show_progress)

    def index(self, document_ids: list[int], texts: list[str]) -> None:
        verify_checkpoint(Path(self.model_name), self.manifest)
        corpus = pd.DataFrame({"answer_id": document_ids, "text": texts})
        ids = corpus_ids(corpus)
        if corpus_fingerprint(corpus) != self.manifest.get("corpus_sha256"):
            raise ValueError("evaluation corpus must match the exported text, IDs and order")
        names = ("answer_ids.npy", f"embeddings_{self.precision}.npy")
        if not verify_export_files(self.artifacts_path, self.manifest, names):
            raise ValueError("precomputed evaluation requires a checksum-verified export")
        candidate = DenseRetriever(self.model_name, device=self.device)
        candidate.load_index(*(self.artifacts_path / name for name in names))
        if not np.array_equal(candidate.document_ids, ids):
            raise ValueError("exported document order does not match the evaluation corpus")
        if candidate.embeddings.shape != (self.manifest["documents"], self.manifest["dimensions"]):
            raise ValueError("exported vectors do not match manifest dimensions")
        expected_dtype = np.float16 if self.precision == "fp16" else np.float32
        if candidate.embeddings.dtype != expected_dtype:
            raise ValueError("exported vector dtype does not match requested precision")
        self.document_ids, self.embeddings = candidate.document_ids, candidate.embeddings
