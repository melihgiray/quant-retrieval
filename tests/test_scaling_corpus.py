import pytest
from scripts.build_scaling_corpus import load_site, main, validate_site


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
