import numpy as np
import pandas as pd
import pytest

from quant_retrieval.serve.snapshot_validation import (
    validate_snapshot_corpus,
    validate_snapshot_vectors,
)


@pytest.mark.parametrize("change, message", [
    ({"documents": 3}, "document count"),
    ({"dimensions": True}, "positive integer"),
    ({"corpus_sha256": "0" * 64}, "fingerprint mismatch"),
    ({"pooling": "cls"}, "mean pooling"),
    ({"commit": "not-a-revision"}, "hexadecimal git revision"),
    ({"commit": True}, "hexadecimal git revision"),
    ({"corpus_sha256": None}, "SHA-256 string"),
    ({"corpus_sha256": ""}, "SHA-256 string"),
])
def test_snapshot_corpus_rejects_inconsistent_metadata(tmp_path, change, message):
    path = tmp_path / "corpus.parquet"
    pd.DataFrame({"answer_id": [1, 2], "question_id": [10, 20],
                  "text": ["bond", "equity"]}).to_parquet(path)
    manifest = {"documents": 2, "dimensions": 2, "max_length": 32, **change}
    with pytest.raises(ValueError, match=message):
        validate_snapshot_corpus(path, manifest)


@pytest.mark.parametrize("ids, vectors, dimensions, message", [
    ([2, 1], np.eye(2, dtype=np.float16), 2, "corpus order"),
    ([1, 2], np.eye(2, dtype=np.float32), 2, "float16"),
    ([1, 2], np.zeros((2, 2), dtype=np.float16), 2, "unit normalized"),
    ([1, 2], np.eye(2, dtype=np.float16), 3, "dimensions"),
])
def test_snapshot_vectors_reject_mismatches(tmp_path, ids, vectors, dimensions, message):
    np.save(tmp_path / "answer_ids.npy", np.array(ids, dtype=np.int64))
    np.save(tmp_path / "embeddings_fp16.npy", vectors)
    frame = pd.DataFrame({"answer_id": [1, 2], "text": ["bond", "equity"]})
    with pytest.raises(ValueError, match=message):
        validate_snapshot_vectors(tmp_path, {"dimensions": dimensions}, frame)
