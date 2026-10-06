import numpy as np
import pandas as pd
import pytest
from scripts import export_queries

from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES, checkpoint_hashes
from quant_retrieval.retrieval.index_artifacts import publish_index
from quant_retrieval.retrieval.query_artifacts import load_queries


def prepare_inputs(tmp_path, monkeypatch):
    model, data, index, output = [tmp_path / name for name in ("model", "data", "index", "queries")]
    model.mkdir()
    data.mkdir()
    for name in CHECKPOINT_FILES:
        (model / name).write_text(name)
    publish_index(index, np.array([1, 2]), np.eye(2), {"max_length": 64, "pooling": "cls",
                  "checkpoint": str(model), "checkpoint_sha256": checkpoint_hashes(model)})
    pd.DataFrame({"question_id": [20, 10, 99], "text": ["twenty", "ten", "held out"],
                  "split": ["val", "val", "test"]}).to_parquet(data / "queries.parquet")
    monkeypatch.setattr("sys.argv", ["export", "--checkpoint", str(model), "--index", str(index),
                                    "--data", str(data), "--output", str(output)])
    return model, output


def test_query_export_only_encodes_selected_validation_text(tmp_path, monkeypatch):
    _, output = prepare_inputs(tmp_path, monkeypatch)
    calls = []

    def encode(checkpoint, texts, metadata, batch_size, device):
        calls.append(texts)
        assert metadata["pooling"] == "cls" and metadata["max_length"] == 64
        return np.eye(2, dtype=np.float32), "cpu"

    monkeypatch.setattr(export_queries, "encode_queries", encode)
    export_queries.main()
    metadata, vectors = load_queries(output)
    assert set(calls[0]) == {"ten", "twenty"}
    assert set(metadata["question_ids"]) == {10, 20}
    assert vectors.shape == (2, 2)
    with pytest.raises(SystemExit):
        export_queries.main()
    assert len(calls) == 1


def test_query_export_detects_checkpoint_change_during_encoding(tmp_path, monkeypatch):
    model, output = prepare_inputs(tmp_path, monkeypatch)

    def encode(*args):
        (model / "model.safetensors").write_text("different weights")
        return np.eye(2, dtype=np.float32), "cpu"

    monkeypatch.setattr(export_queries, "encode_queries", encode)
    with pytest.raises(ValueError, match="checksum mismatch"):
        export_queries.main()
    assert not output.exists()
