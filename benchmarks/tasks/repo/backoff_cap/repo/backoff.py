"""Exponential backoff delays for retries."""

from __future__ import annotations


def delay_for_attempt(attempt: int, *, base: float = 0.5, factor: float = 2.0, max_delay: float = 30.0) -> float:
    """Seconds to wait before retry number `attempt` (0-based): base * factor**attempt, capped at max_delay."""
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    return base * (factor**attempt)
