"""Word-wrap text to a maximum line width, breaking on spaces."""

from __future__ import annotations


def word_wrap(text: str, width: int) -> list[str]:
    """Return `text` split into lines no longer than `width` characters."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= width + 1:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
