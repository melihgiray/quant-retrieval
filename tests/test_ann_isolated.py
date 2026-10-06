"""Real native search in its own process, including on macOS."""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from quant_retrieval.eval.ann_analysis import analyze_ann_report
from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES
from quant_retrieval.retrieval.index_artifacts import publish_index
from quant_retrieval.retrieval.query_artifacts import publish_queries


@pytest.mark.skipif(
    importlib.util.find_spec("faiss") is None, reason="optional FAISS not installed"
)
def test_real_faiss_cached_sweep_runs_without_importing_torch(tmp_path):
    index, cache, output = [tmp_path / name for name in ("index", "queries", "report.json")]
    identity = {name: "a" * 64 for name in CHECKPOINT_FILES}
    publish_index(index, np.array([10, 20, 30]), np.eye(3), {
        "checkpoint": "/fixture/no-model", "checkpoint_sha256": identity,
        "max_length": 64, "pooling": "mean",
    })
    publish_queries(cache, np.eye(3, dtype=np.float32)[:2], {
        "seed": 17, "question_ids": [1, 2], "query_sha256": "b" * 64,
        "split": "val", "checkpoint_sha256": identity, "max_length": 64, "pooling": "mean",
    })
    result = subprocess.run([
        sys.executable, "-c",
        "import runpy, sys; runpy.run_module('scripts.ann_sweep', run_name='__main__'); "
        "assert 'faiss' in sys.modules; assert 'torch' not in sys.modules",
        "--embeddings", str(index), "--query-cache", str(cache), "--output", str(output),
        "--k", "1", "--warmup", "1", "--repeats", "2", "--ef-search", "16", "32",
        "--threads", "1",
    ], check=True, capture_output=True, text=True, timeout=45)
    assert "wrote" in result.stdout
    report = json.loads(output.read_text())
    assert report["complete"] is True
    assert len(report["runs"]) == 3
    assert all(row["recall_at_k"] == 1 and row["samples"] == 4 for row in report["runs"])
    assert analyze_ann_report(report)["artifacts"][0]["documents"] == 3
    assert isinstance(report["ann_runtime"]["faiss"], str)
    report["query_source"]["manifest"]["question_ids"].reverse()
    with pytest.raises(ValueError, match="question_ids"):
        analyze_ann_report(report)
