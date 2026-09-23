import pytest

from inventory import InsufficientStock, Inventory


def test_add_and_remove_within_stock():
    inv = Inventory()
    inv.add("widget", 10)
    inv.remove("widget", 4)
    assert inv.quantity("widget") == 6


def test_non_positive_quantities_are_rejected():
    inv = Inventory()
    with pytest.raises(ValueError):
        inv.add("widget", 0)
    with pytest.raises(ValueError):
        inv.remove("widget", -1)


def test_removing_more_than_on_hand_raises_and_leaves_stock_unchanged():
    inv = Inventory()
    inv.add("widget", 3)
    with pytest.raises(InsufficientStock):
        inv.remove("widget", 5)
    assert inv.quantity("widget") == 3


def test_removing_from_an_unknown_sku_raises():
    inv = Inventory()
    with pytest.raises(InsufficientStock):
        inv.remove("gadget", 1)
