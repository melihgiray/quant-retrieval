import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scripts import download_demo_assets as downloader
from scripts.prepare_demo_snapshot import prepare_demo_snapshot
from scripts.publish_demo_snapshot import publish_demo_snapshot

from quant_retrieval.eval.fingerprints import corpus_fingerprint
from quant_retrieval.retrieval.checkpoint import CHECKPOINT_FILES, checkpoint_hashes
from quant_retrieval.retrieval.index_artifacts import publish_index


def test_versioned_snapshot_roundtrip_with_offline_inspection(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in (*CHECKPOINT_FILES, "added_tokens.json"):
        (checkpoint / name).write_text("{}")
    corpus = pd.DataFrame({"answer_id": [1, 2], "question_id": [10, 20],
                           "text": ["bond", "equity"]})
    corpus_path = tmp_path / "corpus.parquet"
    corpus.to_parquet(corpus_path)
    index = tmp_path / "index"
    publish_index(index, np.array([1, 2]), np.eye(2, dtype=np.float32), {
        "max_length": 32, "checkpoint_sha256": checkpoint_hashes(checkpoint),
        "corpus_sha256": corpus_fingerprint(corpus),
    })
    snapshot = prepare_demo_snapshot(checkpoint, corpus_path, index, tmp_path / "snapshot")
    remote = {}

    def upload(**kwargs):
        remote.update({name: (snapshot / name).read_bytes() for name in kwargs["allow_patterns"]})
        return "fixture-commit"

    api = SimpleNamespace(create_repo=lambda **kw: None, upload_folder=upload,
                          repo_info=lambda **kw: SimpleNamespace(private=True))
    assert publish_demo_snapshot("owner/model", snapshot, api=api) == "fixture-commit"

    def download(**kwargs):
        root = Path(kwargs["local_dir"])
        for name, contents in remote.items():
            assert name in kwargs["allow_patterns"]
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        return str(root)

    monkeypatch.setattr(downloader, "snapshot_download", download)
    local = downloader.download_demo_assets("owner/model", tmp_path / "download", "a" * 40)
    report_path = tmp_path / "inspection.json"
    subprocess.run([sys.executable, "-c",
                    "import runpy,sys; runpy.run_module('scripts.inspect_snapshot', "
                    "run_name='__main__'); assert 'torch' not in sys.modules; "
                    "assert 'transformers' not in sys.modules",
                    "--snapshot", str(local), "--output", str(report_path)], check=True)
    report = json.loads(report_path.read_text())
    assert report["checkpoint_identity_verified"] is True
    assert report["corpus_identity_verified"] is True
    assert report["export_identity_verified"] is True
    assert report["runtime_tested"] is False
