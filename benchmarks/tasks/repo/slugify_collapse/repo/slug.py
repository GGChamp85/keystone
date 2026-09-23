"""URL slugs from free text."""

from __future__ import annotations

import re

_NOT_ALLOWED = re.compile(r"[^a-z0-9]")


def slugify(text: str) -> str:
    lowered = text.strip().lower()
    return _NOT_ALLOWED.sub("-", lowered)
