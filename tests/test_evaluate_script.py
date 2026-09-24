import pytest
import torch
from scripts import evaluate
from scripts.evaluate import build_retriever, set_seed, validate_config

from quant_retrieval.retrieval.bm25 import BM25Retriever
from quant_retrieval.retrieval.dense import DenseRetriever


@pytest.mark.parametrize("key,value", [
    ("run_name", "../result"), ("seed", True), ("seed", -1), ("seed", 2**32),
    ("split", "validation"), ("max_results", 99), ("max_results", 100.5),
    ("parameters", []), ("output", ""), ("retriever", "unknown"),
])
def test_evaluation_config_rejects_invalid_settings_before_model_work(key, value):
    config = {"run_name": "test_run", "seed": 17, "retriever": "bm25", key: value}
    with pytest.raises(ValueError):
        validate_config(config)


def test_evaluation_config_defaults_and_empty_documents():
    validate_config({"run_name": "tiny_run", "seed": 0, "retriever": "bm25"})
    with pytest.raises(ValueError, match="object"):
        validate_config(None)


def test_existing_evaluation_output_is_rejected_before_model_work(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("run_name: tiny\nseed: 17\nretriever: bm25\n")
    output = tmp_path / "result.json"
    output.write_text("existing result")
    monkeypatch.setattr("sys.argv", ["evaluate", "--config", str(config),
                                    "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    assert output.read_text() == "existing result"


@pytest.mark.parametrize("allow_test", [False, True])
def test_test_split_requires_explicit_permission_before_model_loading(
    tmp_path, monkeypatch, allow_test
):
    path = tmp_path / "final.yaml"
    path.write_text("run_name: final\nseed: 17\nretriever: bm25\nsplit: test\n")
    monkeypatch.setattr(evaluate, "set_seed", lambda seed: None)

    def reached_model_boundary(config):
        raise RuntimeError("model boundary reached; no data loaded")

    monkeypatch.setattr(evaluate, "build_retriever", reached_model_boundary)
    argv = ["evaluate", "--config", str(path)] + (["--allow-test"] if allow_test else [])
    monkeypatch.setattr("sys.argv", argv)
    if allow_test:
        with pytest.raises(RuntimeError, match="model boundary"):
            evaluate.main()
    else:
        with pytest.raises(SystemExit) as error:
            evaluate.main()
        assert error.value.code == 2


def test_builds_bm25_from_config_parameters():
    retriever = build_retriever(
        {"retriever": "bm25", "parameters": {"k1": 1.6, "b": 0.4}}
    )
    assert isinstance(retriever, BM25Retriever)
    assert retriever.k1 == 1.6
    assert retriever.b == 0.4


def test_rejects_unknown_retrievers():
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever({"retriever": "magic"})


def test_builds_dense_retriever_without_loading_a_model():
    retriever = build_retriever(
        {
            "retriever": "dense",
            "parameters": {
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "batch_size": 32,
                "device": "cpu",
            },
        }
    )
    assert isinstance(retriever, DenseRetriever)
    assert retriever.batch_size == 32
    assert retriever.device == "cpu"
    assert retriever._model is None


def test_seed_repeats_torch_random_values():
    set_seed(17)
    first = torch.rand(4)
    set_seed(17)
    assert torch.equal(first, torch.rand(4))


def test_hybrid_specs_build_their_children():
    retriever = build_retriever(
        {
            "retriever": "hybrid",
            "parameters": {
                "rrf_k": 30,
                "retrievers": [
                    {"retriever": "bm25", "parameters": {"k1": 1.2}},
                    {"retriever": "bm25", "parameters": {"k1": 0.9}},
                ],
            },
        }
    )
    assert retriever.rrf_k == 30
    assert [r.k1 for r in retriever.retrievers] == [1.2, 0.9]


def test_reranking_specs_build_their_base():
    retriever = build_retriever(
        {
            "retriever": "rerank",
            "parameters": {
                "model_name": "checkpoints/nowhere",
                "depth": 25,
                "base": {"retriever": "bm25", "parameters": {"k1": 1.4}},
            },
        }
    )
    assert retriever.depth == 25
    assert retriever.base.k1 == 1.4


def test_a_pipeline_can_nest_more_than_one_level():
    retriever = build_retriever(
        {
            "retriever": "rerank",
            "parameters": {
                "model_name": "checkpoints/nowhere",
                "base": {
                    "retriever": "hybrid",
                    "parameters": {
                        "retrievers": [
                            {"retriever": "bm25", "parameters": {}},
                            {"retriever": "bm25", "parameters": {}},
                        ]
                    },
                },
            },
        }
    )
    assert len(retriever.base.retrievers) == 2


def test_building_a_spec_does_not_consume_it():
    # The spec is written into the committed result as provenance, so building
    # from it must not pop keys out of the caller's dictionary.
    spec = {
        "retriever": "hybrid",
        "parameters": {
            "retrievers": [
                {"retriever": "bm25", "parameters": {}},
                {"retriever": "bm25", "parameters": {}},
            ]
        },
    }
    build_retriever(spec)
    assert "retrievers" in spec["parameters"]
