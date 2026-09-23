# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — structure-aware code chunking for retrieval.

A chunk that starts mid-function and ends mid-class retrieves badly: the embedding describes half of two
things. This module cuts on the code's own boundaries instead of every N characters:

- **Python**: the real `ast` — every top-level function, async function and class (with decorators and the
  docstring) is one unit; small consecutive units are folded together up to `chunk_size`; a unit larger
  than `chunk_size` is split by the line-based fallback, so nothing is dropped. Module-level code between
  definitions stays with its neighbours in order.
- **Go, JavaScript/TypeScript, Rust, Java, C#, Ruby, PHP**: a line-boundary scan for the language's
  definition keywords at low indentation (`func`, `function`/`class`/`export`, `fn`/`impl`, ...). Cheaper
  than a parser and wrong less often than a fixed window; where the sandbox's ctags is a better source
  (the agent's repository map) the units line up with what the model already saw.
- **Anything else**: the original line-based window with overlap.

`chunk_code` never returns an empty list for non-empty input and never loses text: joining the chunks
(minus overlap) reproduces the file. Pure, unit-tested (`tests/test_chunking.py`).
"""

from __future__ import annotations

import ast
import re

_DEF_PATTERNS: dict[str, re.Pattern[str]] = {
    "go": re.compile(r"^(func|type)\s"),
    "javascript": re.compile(
        r"^(export\s+)?(async\s+)?(function|class)\s|^(export\s+)?(const|let|var)\s+\w+\s*=\s*(async\s*)?\("
    ),
    "typescript": re.compile(
        r"^(export\s+)?(default\s+)?(async\s+)?(function|class|interface|type|enum)\s|"
        r"^(export\s+)?(const|let|var)\s+\w+\s*(:[^=]+)?=\s*(async\s*)?\("
    ),
    "rust": re.compile(r"^(pub(\([^)]*\))?\s+)?(async\s+)?(fn|impl|struct|enum|trait|mod)\s"),
    "java": re.compile(
        r"^\s{0,4}(public|private|protected|static|final|abstract|\s)*\s*(class|interface|enum|record)\s|"
        r"^\s{4}(public|private|protected)\s[^=;]*\("
    ),
    "csharp": re.compile(
        r"^\s{0,4}(public|private|protected|internal|static|\s)*\s*(class|interface|enum|struct|record)\s|"
        r"^\s{4,8}(public|private|protected|internal)\s[^=;]*\("
    ),
    "ruby": re.compile(r"^\s{0,2}(def|class|module)\s"),
    "php": re.compile(
        r"^\s{0,4}(abstract\s+|final\s+)?(class|interface|trait|function)\s|"
        r"^\s{4}(public|private|protected)\s+(static\s+)?function\s"
    ),
}


def line_chunks(code: str, chunk_size: int, overlap: int) -> list[str]:
    """The fixed window: lines packed up to `chunk_size` characters, the last `overlap` characters of lines
    repeated at the start of the next chunk."""
    if len(code) <= chunk_size:
        return [code]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in code.split("\n"):
        line_len = len(line) + 1
        if current_len + line_len > chunk_size and current:
            chunks.append("\n".join(current))
            overlap_lines: list[str] = []
            overlap_len = 0
            for prev in reversed(current):
                if overlap_len + len(prev) + 1 > overlap:
                    break
                overlap_lines.insert(0, prev)
                overlap_len += len(prev) + 1
            current = overlap_lines
            current_len = overlap_len
        current.append(line)
        current_len += line_len
    if current:
        chunks.append("\n".join(current))
    return chunks


def _python_units(code: str) -> list[str] | None:
    """Top-level definitions (with decorators) and the module-level code between them, in order.
    None when the file does not parse (a template, Python 2, a broken file) — the caller falls back."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    lines = code.split("\n")
    units: list[str] = []
    cursor = 0  # next unclaimed line index
    for node in tree.body:
        decorators = getattr(node, "decorator_list", []) or []
        start = (min(d.lineno for d in decorators) if decorators else node.lineno) - 1
        end = node.end_lineno or node.lineno
        if start > cursor:
            between = "\n".join(lines[cursor:start]).strip("\n")
            if between.strip():
                units.append(between)
        units.append("\n".join(lines[start:end]))
        cursor = end
    if cursor < len(lines):
        tail = "\n".join(lines[cursor:]).strip("\n")
        if tail.strip():
            units.append(tail)
    return units or None


def _keyword_units(code: str, pattern: re.Pattern[str]) -> list[str] | None:
    lines = code.split("\n")
    starts = [i for i, line in enumerate(lines) if pattern.match(line)]
    if len(starts) < 2:
        return None
    units: list[str] = []
    if starts[0] > 0:
        head = "\n".join(lines[: starts[0]]).strip("\n")
        if head.strip():
            units.append(head)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(lines)
        units.append("\n".join(lines[s:e]).rstrip("\n"))
    return units


def _pack(units: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Fold consecutive small units up to `chunk_size`; split a unit larger than `chunk_size` by lines."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_len = [], 0
            chunks.extend(line_chunks(unit, chunk_size, overlap))
            continue
        if current and current_len + len(unit) + 2 > chunk_size:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(unit)
        current_len += len(unit) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_code(code: str, language: str | None, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    """Structure-aware chunks for `language` (the ingestion's LANGUAGE_MAP names), falling back to the
    line window. Every chunk is non-empty; short files are one chunk."""
    if not code.strip():
        return [code] if code else []
    if len(code) <= chunk_size:
        return [code]
    units: list[str] | None = None
    if language == "python":
        units = _python_units(code)
    elif language in _DEF_PATTERNS:
        units = _keyword_units(code, _DEF_PATTERNS[language])
    if not units:
        return line_chunks(code, chunk_size, overlap)
    return [c for c in _pack(units, chunk_size, overlap) if c.strip()]
