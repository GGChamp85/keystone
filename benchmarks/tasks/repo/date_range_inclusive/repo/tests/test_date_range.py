from datetime import date

import pytest

from date_range import dates_between


def test_reversed_range_is_rejected():
    with pytest.raises(ValueError):
        dates_between(date(2026, 1, 5), date(2026, 1, 1))


def test_range_starts_at_start():
    assert dates_between(date(2026, 1, 1), date(2026, 1, 3))[0] == date(2026, 1, 1)


def test_range_includes_the_end_date():
    assert dates_between(date(2026, 1, 1), date(2026, 1, 3)) == [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]


def test_single_day_range_has_one_date():
    assert dates_between(date(2026, 2, 28), date(2026, 2, 28)) == [date(2026, 2, 28)]
