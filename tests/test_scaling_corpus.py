import pandas as pd
import pytest
from scripts.build_scaling_corpus import (
    ID_BLOCK_SIZE,
    combine_sources,
    load_site,
    main,
    namespace_corpus,
    validate_site,
)


@pytest.mark.parametrize("site", ["../stats.stackexchange.com", "https://stats.stackexchange.com",
                                 "stats.stackexchange.com/extra", "stats.example.com", "", "Stats"])
def test_invalid_sites_fail_before_download(site, tmp_path):
    with pytest.raises(ValueError, match="hostname"):
        load_site(site, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_valid_stack_exchange_hostnames():
    assert validate_site("stats.stackexchange.com") == "stats.stackexchange.com"
    assert validate_site("meta.physics.stackexchange.com") == "meta.physics.stackexchange.com"


@pytest.mark.parametrize("arguments", [["--sizes", "0"],
    ["--sites", "stats.stackexchange.com", "stats.stackexchange.com"]])
def test_scaling_arguments_fail_before_reading_data(monkeypatch, arguments):
    monkeypatch.setattr("sys.argv", ["scaling", *arguments])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_namespace_preserves_source_and_rejects_block_overflow():
    source = pd.DataFrame({"answer_id": [1, 2], "question_id": [3, 3]})
    namespaced = namespace_corpus(source, "stats.stackexchange.com")
    assert source.answer_id.tolist() == [1, 2]
    assert namespaced.answer_id.iloc[1] - namespaced.answer_id.iloc[0] == 1
    source.loc[0, "answer_id"] = ID_BLOCK_SIZE
    with pytest.raises(ValueError, match="reserved ID block"):
        namespace_corpus(source, "stats.stackexchange.com")


@pytest.mark.parametrize("extra_ids", [[1, 3], [3, 3]])
def test_collisions_are_not_silently_dropped(extra_ids):
    base = pd.DataFrame({"answer_id": [1, 2]})
    with pytest.raises(ValueError, match="collide"):
        combine_sources(base, [pd.DataFrame({"answer_id": extra_ids})])
