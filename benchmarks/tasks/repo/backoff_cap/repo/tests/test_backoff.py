import pytest

from backoff import delay_for_attempt


def test_first_attempts_grow_exponentially():
    assert delay_for_attempt(0) == 0.5
    assert delay_for_attempt(1) == 1.0
    assert delay_for_attempt(2) == 2.0


def test_negative_attempt_is_rejected():
    with pytest.raises(ValueError):
        delay_for_attempt(-1)


def test_delay_is_capped_at_max_delay():
    assert delay_for_attempt(10) == 30.0
    assert delay_for_attempt(6, max_delay=10.0) == 10.0


def test_cap_does_not_affect_small_delays():
    assert delay_for_attempt(3, max_delay=10.0) == 4.0
