from copy import deepcopy

import pytest

from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES
from quant_retrieval.retrieval.query_artifacts import validate_query_manifest


def query_metadata():
    return {"schema_version": 1, "kind": "ann_queries", "split": "val", "queries": 2,
            "dimensions": 2, "max_length": 64, "pooling": "mean", "dtype": "float32",
            "seed": 17, "question_ids": [20, 10], "query_sha256": "a" * 64,
            "vectors_sha256": "b" * 64,
            "checkpoint_sha256": {name: "c" * 64 for name in CHECKPOINT_FILES}}


def test_query_contract_keeps_sample_order_and_metadata_unchanged():
    metadata = query_metadata()
    before = deepcopy(metadata)
    validate_query_manifest(metadata)
    assert metadata == before


@pytest.mark.parametrize("key,value", [
    ("split", "test"), ("split", "train"), ("schema_version", True), ("kind", "documents"),
    ("question_ids", [20, 20]), ("question_ids", [2**63, 1]), ("question_ids", [True, 10]),
    ("queries", 3), ("dimensions", 0), ("seed", -1), ("dtype", "float16"),
    ("pooling", []), ("query_sha256", "unknown"), ("checkpoint_sha256", None),
])
def test_query_contract_rejects_ambiguous_or_incompatible_metadata(key, value):
    metadata = query_metadata()
    metadata[key] = value
    with pytest.raises(ValueError):
        validate_query_manifest(metadata)
