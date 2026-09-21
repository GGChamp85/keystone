# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real, no-mock tests for src/finetuning/data_prep.py — real local files on
disk, real JSONL written and read back. No prior test file covered this
module at all before this one.
"""

from __future__ import annotations

import json

from src.finetuning.data_prep import (
    prepare_continued_pretraining_data,
    prepare_dpo_data,
    prepare_sft_data,
)


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_pretraining_data_includes_real_file_content(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "main.py", "def add(a, b):\n    return a + b\n" * 3)  # well over the 50-char minimum
    out = tmp_path / "out.jsonl"

    stats = prepare_continued_pretraining_data([str(repo)], str(out))

    assert stats["files_processed"] == 1
    assert stats["records"] == 1
    records = _read_jsonl(out)
    assert "def add(a, b):" in records[0]["text"]
    assert records[0]["file_path"] == "main.py"


def test_pretraining_data_dedupes_identical_content_across_files():
    """The real gap this fixed: sha256 was computed per file but never
    checked against content already seen — a vendored/duplicated file
    would train on the exact same text more than once."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo = root / "repo"
        content = "def shared():\n    return 42\n" * 5
        _write(repo / "a.py", content)
        _write(repo / "vendored" / "a_copy.py", content)  # byte-identical content, different path
        _write(repo / "b.py", "def different():\n    return 1\n" * 5)
        out = root / "out.jsonl"

        stats = prepare_continued_pretraining_data([str(repo)], str(out))

        assert stats["files_deduped"] == 1
        assert stats["records"] == 2  # a.py (first seen) + b.py, not a_copy.py
        records = _read_jsonl(out)
        paths = {r["file_path"] for r in records}
        assert paths == {"a.py", "b.py"}


def test_pretraining_data_skips_tiny_files(tmp_path):
    repo = tmp_path / "repo"
    _write(repo / "empty.py", "x=1")  # under the 50-char minimum
    out = tmp_path / "out.jsonl"

    stats = prepare_continued_pretraining_data([str(repo)], str(out))
    assert stats["records"] == 0
    assert stats["files_skipped"] == 1


def test_pretraining_data_skips_a_nonexistent_repo_path(tmp_path):
    out = tmp_path / "out.jsonl"
    stats = prepare_continued_pretraining_data([str(tmp_path / "does-not-exist")], str(out))
    assert stats["records"] == 0


def test_sft_data_combines_public_pairs_and_internal_tasks(tmp_path):
    out = tmp_path / "sft.jsonl"
    stats = prepare_sft_data(
        instruction_pairs=[{"instruction": "Write a hello world function", "input": "", "output": "def hello(): ..."}],
        internal_tasks=[{"task": "Fix the off-by-one bug", "context": "loop bounds", "solution": "range(n)"}],
        output_path=str(out),
    )
    assert stats["total_records"] == 2
    records = _read_jsonl(out)
    assert records[0]["messages"][1]["content"] == "Write a hello world function"
    assert records[0]["messages"][2]["content"] == "def hello(): ..."
    assert "Fix the off-by-one bug" in records[1]["messages"][1]["content"]


def test_dpo_data_writes_real_prompt_chosen_rejected_shape(tmp_path):
    out = tmp_path / "dpo.jsonl"
    stats = prepare_dpo_data(
        preference_pairs=[{"prompt": "Add input validation", "chosen": "def f(x): ...", "rejected": "def f(x): pass"}],
        output_path=str(out),
    )
    assert stats["total_pairs"] == 1
    records = _read_jsonl(out)
    assert records[0]["prompt"][1]["content"] == "Add input validation"
    assert records[0]["chosen"][0]["content"] == "def f(x): ..."
    assert records[0]["rejected"][0]["content"] == "def f(x): pass"
