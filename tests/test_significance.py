import numpy as np
import pytest

from quant_retrieval.eval.significance import format_difference, paired_bootstrap


def scores(values):
    return {index: value for index, value in enumerate(values)}


def test_identical_runs_show_no_difference_and_no_significance():
    same = scores([0.1, 0.5, 0.9, 0.3, 0.7] * 20)
    result = paired_bootstrap(same, dict(same), iterations=2000)

    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["ci_low"] == pytest.approx(0.0)
    assert result["ci_high"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)
    assert result["significant"] is False


def test_a_shift_larger_than_the_spread_is_significant():
    # Every question improves by 0.1, so no resample can make the difference
    # vanish and the interval cannot contain zero.
    baseline = scores([0.1, 0.5, 0.9, 0.3, 0.7] * 20)
    candidate = {query: value + 0.1 for query, value in baseline.items()}

    result = paired_bootstrap(baseline, candidate, iterations=2000)

    assert result["mean_difference"] == pytest.approx(0.1)
    assert result["ci_low"] > 0
    assert result["significant"] is True
    assert result["p_value"] < 0.01


def test_a_tiny_difference_swamped_by_noise_is_not_significant():
    generator = np.random.default_rng(0)
    baseline = scores(generator.random(400))
    # Half the questions gain a lot, half lose almost as much. The average moves
    # barely at all and the interval should straddle zero.
    candidate = {
        query: value + (0.4 if query % 2 else -0.39) for query, value in baseline.items()
    }

    result = paired_bootstrap(baseline, candidate, iterations=2000)

    assert abs(result["mean_difference"]) < 0.01
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["significant"] is False


def test_pairing_finds_a_small_consistent_gain_that_noise_would_hide():
    # Question difficulty varies hugely, the gain is small and consistent.
    # Comparing unpaired samples would drown it; pairing cancels the difficulty.
    generator = np.random.default_rng(1)
    baseline = scores(generator.random(500))
    candidate = {query: value + 0.02 for query, value in baseline.items()}

    result = paired_bootstrap(baseline, candidate, iterations=2000)

    assert result["significant"] is True
    assert result["ci_low"] > 0


def test_the_interval_brackets_the_observed_difference():
    generator = np.random.default_rng(2)
    baseline = scores(generator.random(200))
    candidate = {query: value + generator.normal(0.05, 0.1) for query, value in baseline.items()}

    result = paired_bootstrap(baseline, candidate, iterations=2000)

    assert result["ci_low"] <= result["mean_difference"] <= result["ci_high"]


def test_the_same_seed_gives_the_same_answer():
    generator = np.random.default_rng(3)
    baseline = scores(generator.random(100))
    candidate = {query: value + generator.normal(0, 0.1) for query, value in baseline.items()}

    first = paired_bootstrap(baseline, candidate, iterations=500, seed=7)
    second = paired_bootstrap(baseline, candidate, iterations=500, seed=7)
    third = paired_bootstrap(baseline, candidate, iterations=500, seed=8)

    assert first == second
    assert first["ci_low"] != third["ci_low"]


def test_a_wider_confidence_level_gives_a_wider_interval():
    generator = np.random.default_rng(4)
    baseline = scores(generator.random(200))
    candidate = {query: value + generator.normal(0.05, 0.2) for query, value in baseline.items()}

    narrow = paired_bootstrap(baseline, candidate, iterations=2000, confidence=0.80)
    wide = paired_bootstrap(baseline, candidate, iterations=2000, confidence=0.99)

    assert wide["ci_low"] < narrow["ci_low"]
    assert wide["ci_high"] > narrow["ci_high"]


def test_runs_covering_different_queries_are_rejected():
    with pytest.raises(ValueError, match="different queries"):
        paired_bootstrap(scores([0.1, 0.2, 0.3]), scores([0.1, 0.2]))


def test_runs_sharing_nothing_are_rejected():
    with pytest.raises(ValueError, match="share no queries"):
        paired_bootstrap({1: 0.5}, {2: 0.5})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"iterations": 0}, "iterations"), ({"confidence": 1.0}, "confidence")],
)
def test_invalid_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(scores([0.1, 0.2]), scores([0.2, 0.3]), **kwargs)


def test_format_difference_reads_as_a_table_cell():
    line = format_difference(
        {"mean_difference": 0.0396, "ci_low": 0.021, "ci_high": 0.058, "p_value": 0.0004}
    )
    assert line == "+0.0396 [+0.0210, +0.0580], p = 0.000"
