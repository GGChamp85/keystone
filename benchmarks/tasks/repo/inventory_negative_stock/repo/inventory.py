"""Warehouse stock levels."""

from __future__ import annotations


class InsufficientStock(Exception):
    """Raised when a removal would take a SKU's quantity below zero."""


class Inventory:
    def __init__(self) -> None:
        self._stock: dict[str, int] = {}

    def add(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + quantity

    def remove(self, sku: str, quantity: int) -> None:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._stock[sku] = self._stock.get(sku, 0) - quantity

    def quantity(self, sku: str) -> int:
        return self._stock.get(sku, 0)
