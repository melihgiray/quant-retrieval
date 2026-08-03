import pytest

from quant_retrieval.models.schedule import linear_warmup_decay


def test_warmup_climbs_to_the_full_rate():
    # Ten warmup steps, so step 0 runs at a tenth and step 9 at the full rate.
    assert linear_warmup_decay(0, 100, 10) == pytest.approx(0.1)
    assert linear_warmup_decay(4, 100, 10) == pytest.approx(0.5)
    assert linear_warmup_decay(9, 100, 10) == pytest.approx(1.0)


def test_decay_falls_linearly_to_zero():
    assert linear_warmup_decay(10, 100, 10) == pytest.approx(1.0)
    assert linear_warmup_decay(55, 100, 10) == pytest.approx(0.5)
    assert linear_warmup_decay(100, 100, 10) == pytest.approx(0.0)


def test_the_rate_never_goes_negative_past_the_end():
    assert linear_warmup_decay(120, 100, 10) == 0.0


def test_no_warmup_starts_at_the_full_rate():
    assert linear_warmup_decay(0, 100, 0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("total", "warmup", "message"),
    [(0, 0, "total_steps"), (100, -1, "warmup_steps"), (100, 101, "warmup_steps")],
)
def test_invalid_schedules_are_rejected(total, warmup, message):
    with pytest.raises(ValueError, match=message):
        linear_warmup_decay(0, total, warmup)
