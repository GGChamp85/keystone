import pytest

from pagination import page_bounds, page_count


def test_exact_multiple():
    assert page_count(10, 5) == 2


def test_zero_items_zero_pages():
    assert page_count(0, 5) == 0


def test_invalid_arguments():
    with pytest.raises(ValueError):
        page_count(10, 0)
    with pytest.raises(ValueError):
        page_count(-1, 5)


def test_partial_last_page_is_counted():
    assert page_count(11, 5) == 3
    assert page_count(1, 5) == 1


def test_bounds_of_the_partial_last_page():
    assert page_bounds(3, 5, 11) == (10, 11)
