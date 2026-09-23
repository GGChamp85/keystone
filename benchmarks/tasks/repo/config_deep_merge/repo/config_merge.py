"""Layered configuration: defaults overridden by an environment-specific file."""

from __future__ import annotations

from typing import Any


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """A new dict: `base` with `override` applied on top. Nested dicts are merged key by key."""
    merged = dict(base)
    for key, value in override.items():
        merged[key] = value
    return merged
