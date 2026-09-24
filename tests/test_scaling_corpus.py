import json

import pandas as pd
import pytest
from scripts import build_scaling_corpus
from scripts.build_scaling_corpus import (
    ID_BLOCK_SIZE,
    combine_sources,
    file_digest,
    load_site,
    main,
    namespace_corpus,
    nested_corpora,
    validate_site,
    write_corpus,
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


def test_nested_corpora_keep_gold_answers_and_share_distractor_prefixes():
    base = pd.DataFrame({"answer_id": [2, 1], "text": ["b", "a"]})
    pool = pd.DataFrame({"answer_id": range(3, 30), "text": ["extra"] * 27})
    first = dict(nested_corpora(base, pool, [9, 4, 9], 17))
    second = dict(nested_corpora(base, pool.iloc[::-1], [4, 9], 17))
    assert list(first) == [4, 9]
    pd.testing.assert_frame_equal(first[4], first[9].head(4))
    pd.testing.assert_frame_equal(first[9].head(2), base)
    pd.testing.assert_frame_equal(first[9], second[9])
    changed = dict(nested_corpora(base, pool, [9], 18))[9]
    assert first[9].answer_id.tolist() != changed.answer_id.tolist()


@pytest.mark.parametrize("sizes", [[1], [5], [2, 5], []])
def test_impossible_size_stops_before_yielding_any_corpus(sizes):
    base = pd.DataFrame({"answer_id": [1, 2]})
    pool = pd.DataFrame({"answer_id": [3, 4]})
    with pytest.raises(ValueError):
        next(nested_corpora(base, pool, sizes, 17))


def test_atomic_corpus_write_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "corpus.parquet"
    original = pd.DataFrame({"answer_id": [1]})
    write_corpus(original, path)

    def fail(self, temporary, **kwargs):
        temporary.write_bytes(b"partial parquet")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="disk full"):
        write_corpus(pd.DataFrame({"answer_id": [2]}), path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)
    assert list(tmp_path.iterdir()) == [path]


def test_scaling_cli_records_source_and_output_identity(tmp_path, monkeypatch):
    base = pd.DataFrame({"answer_id": [1, 2], "text": ["base one", "base two"]})
    base.to_parquet(tmp_path / "corpus.parquet")

    def fake_site(site, raw_root):
        directory = raw_root / site
        directory.mkdir(parents=True)
        (directory / "Posts.xml").write_text("fixture source")
        return pd.DataFrame({"answer_id": [10, 11], "text": ["extra one", "extra two"]})

    monkeypatch.setattr(build_scaling_corpus, "load_site", fake_site)
    monkeypatch.setattr("sys.argv", ["scaling", "--data", str(tmp_path),
                                    "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out"),
                                    "--sites", "stats.stackexchange.com", "--sizes", "3", "4"])
    main()
    output = tmp_path / "out"
    summary = json.loads((output / "scaling_corpora.json").read_text())
    assert summary["complete"] is True
    assert summary["seed"] == 17
    assert summary["base_sha256"] == file_digest(tmp_path / "corpus.parquet")
    assert summary["sha256"]["4"] == file_digest(output / "scaling_corpus_4.parquet")
    assert summary["sources"][0]["id_offset"] > 0
    largest = pd.read_parquet(output / "scaling_corpus_4.parquet")
    pd.testing.assert_frame_equal(largest.head(2), base)
