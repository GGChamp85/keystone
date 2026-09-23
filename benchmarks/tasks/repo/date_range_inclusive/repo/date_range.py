"""Calendar date ranges."""

from __future__ import annotations

from datetime import date, timedelta


def dates_between(start: date, end: date) -> list[date]:
    """Every date from `start` to `end`, INCLUSIVE of both ends, in order."""
    if end < start:
        raise ValueError("end must not be before start")
    days = (end - start).days
    return [start + timedelta(days=i) for i in range(days)]
