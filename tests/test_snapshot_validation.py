import pandas as pd
import pytest

from quant_retrieval.serve.snapshot_validation import validate_snapshot_corpus


@pytest.mark.parametrize("change, message", [
    ({"documents": 3}, "document count"),
    ({"dimensions": True}, "positive integer"),
    ({"corpus_sha256": "0" * 64}, "fingerprint mismatch"),
    ({"pooling": "cls"}, "mean pooling"),
])
def test_snapshot_corpus_rejects_inconsistent_metadata(tmp_path, change, message):
    path = tmp_path / "corpus.parquet"
    pd.DataFrame({"answer_id": [1, 2], "question_id": [10, 20],
                  "text": ["bond", "equity"]}).to_parquet(path)
    manifest = {"documents": 2, "dimensions": 2, "max_length": 32, **change}
    with pytest.raises(ValueError, match=message):
        validate_snapshot_corpus(path, manifest)
