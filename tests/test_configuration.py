import pytest

from quant_retrieval.configuration import load_config


@pytest.mark.parametrize("contents", ["seed: 17\nseed: 42\n",
    "parameters:\n  depth: 10\n  depth: 50\n", "[one, two]", "", "1: numeric-key"])
def test_ambiguous_or_nonmapping_configurations_fail(tmp_path, contents):
    path = tmp_path / "experiment.yaml"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_config(path)


def test_nested_pipeline_configuration_remains_readable(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("retriever: hybrid\nparameters:\n  retrievers:\n    - retriever: bm25\n")
    assert load_config(path)["parameters"]["retrievers"] == [{"retriever": "bm25"}]


def test_yaml_object_construction_is_not_enabled(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("value: !!python/object:builtins.object {}")
    with pytest.raises(ValueError, match="invalid configuration"):
        load_config(path)
