# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Structure-aware chunking (src/memory/chunking.py): Python cuts on real ast boundaries, other languages
on their definition keywords, everything else on the line window — and no text is ever lost."""

from __future__ import annotations

from src.memory.chunking import chunk_code, line_chunks

PY = '''"""module docstring"""
import os

CONST = 1


def alpha(x):
    """first"""
    return x + 1


@decorator
def beta(y):
    total = 0
    for i in range(y):
        total += i
    return total


class Gamma:
    def method(self):
        return "g"

    def other(self):
        return "o"


def tail():
    return None
'''


def test_python_chunks_start_on_definitions_and_keep_decorators():
    chunks = chunk_code(PY, "python", chunk_size=120, overlap=20)
    assert len(chunks) >= 3
    starts = [c.split("\n", 1)[0] for c in chunks]
    # every chunk after the module header starts on a top-level definition (or its decorator), never mid-body
    for first_line in starts[1:]:
        assert first_line.startswith(("def ", "@", "class ")), first_line
    beta = next(c for c in chunks if "def beta" in c)
    assert beta.startswith("@decorator") and "return total" in beta  # decorator and body stay together
    gamma = next(c for c in chunks if "class Gamma" in c)
    assert "def other" in gamma  # a class is one unit
    assert "".join(chunks).count("def ") == PY.count("def ")  # nothing dropped


def test_small_python_units_are_folded_and_large_ones_split_without_loss():
    big_body = "\n".join(f"    x{i} = {i}" for i in range(200))
    code = f"def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef huge():\n{big_body}\n    return x1\n"
    chunks = chunk_code(code, "python", chunk_size=300, overlap=40)
    assert "def a" in chunks[0] and "def b" in chunks[0]  # folded: two small functions in one chunk
    huge_parts = [c for c in chunks if "x199 = 199" in c or "def huge" in c]
    assert len(huge_parts) >= 2  # the oversized function was split by lines
    assert all(len(c) <= 300 + 80 for c in chunks)  # a line-window chunk may overshoot by one line at most


def test_unparseable_python_falls_back_to_the_line_window():
    broken = "def x(:\n" + "\n".join(f"line {i}" for i in range(100))
    assert chunk_code(broken, "python", chunk_size=200, overlap=20) == line_chunks(broken, 200, 20)


def test_go_and_typescript_cut_on_their_definition_keywords():
    go = 'package main\n\nimport "fmt"\n\n' + "\n\n".join(
        f"func f{i}() {{\n\tfmt.Println({i})\n\treturn\n}}" for i in range(12)
    )
    chunks = chunk_code(go, "go", chunk_size=120, overlap=20)
    assert all(c.startswith(("package", "func ")) for c in chunks)
    ts = "import x from 'y';\n\n" + "\n\n".join(
        f"export async function handler{i}(req: Request) {{\n  return {i};\n}}" for i in range(12)
    )
    tchunks = chunk_code(ts, "typescript", chunk_size=120, overlap=20)
    assert all(c.startswith(("import", "export ")) for c in tchunks)


def test_short_files_are_one_chunk_and_unknown_languages_use_the_window():
    assert chunk_code("tiny", "python") == ["tiny"]
    text = "\n".join(f"row {i}" for i in range(500))
    assert chunk_code(text, "text", chunk_size=100, overlap=10) == line_chunks(text, 100, 10)
    assert chunk_code("", "python") == []
