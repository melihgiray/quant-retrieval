from pathlib import Path

import pytest

from quant_retrieval.serve.settings import ServeSettings


def test_settings_have_runnable_local_defaults():
    settings = ServeSettings.from_environment({})

    assert settings.model_path == Path("checkpoints/minilm_tuned/epoch-3")
    assert settings.manifest_path == Path("artifacts/manifest.json")
    assert settings.embeddings_path == Path("artifacts/embeddings_fp16.npy")
    assert settings.depth == 100
    assert settings.rrf_k == 60


def test_settings_read_deployment_overrides():
    settings = ServeSettings.from_environment(
        {
            "MODEL_PATH": "/model",
            "CORPUS_PATH": "/data/corpus.parquet",
            "MANIFEST_PATH": "/data/manifest.json",
            "DOCUMENT_IDS_PATH": "/data/ids.npy",
            "EMBEDDINGS_PATH": "/data/embeddings.npy",
            "DEVICE": "cpu",
            "RETRIEVAL_DEPTH": "75",
            "RRF_K": "40",
        }
    )

    assert settings.model_path == Path("/model")
    assert settings.manifest_path == Path("/data/manifest.json")
    assert settings.device == "cpu"
    assert settings.depth == 75
    assert settings.rrf_k == 40


@pytest.mark.parametrize("name", ["RETRIEVAL_DEPTH", "RRF_K"])
def test_settings_reject_nonpositive_retrieval_parameters(name):
    with pytest.raises(ValueError, match="positive"):
        ServeSettings.from_environment({name: "0"})
