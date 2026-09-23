"""Page arithmetic for list endpoints."""

from __future__ import annotations


def page_count(total_items: int, page_size: int) -> int:
    """How many pages `total_items` need at `page_size` per page."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if total_items < 0:
        raise ValueError("total_items must not be negative")
    return total_items // page_size


def page_bounds(page: int, page_size: int, total_items: int) -> tuple[int, int]:
    """(start, end) item indexes for 1-based `page`, end exclusive; raises for a page past the last."""
    if page < 1 or page > page_count(total_items, page_size):
        raise IndexError("page out of range")
    start = (page - 1) * page_size
    return start, min(start + page_size, total_items)
