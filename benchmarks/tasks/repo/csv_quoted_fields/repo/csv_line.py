"""A minimal CSV line parser with no dependency on the csv module."""

from __future__ import annotations


def parse_line(line: str) -> list[str]:
    """Split one CSV line into its fields."""
    return [field.strip() for field in line.rstrip("\n").split(",")]
