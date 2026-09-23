"""Flatten a nested dict into dot-path keys, e.g. {"a": {"b": 1}} -> {"a.b": 1}."""

from __future__ import annotations

from typing import Any


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Return a new dict with every nested key path joined by dots."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out
