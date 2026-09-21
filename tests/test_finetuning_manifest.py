# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real, no-mock tests for src/finetuning/manifest.py's train/holdout split and
manifest writing — pure logic and real file I/O, no ML dependencies needed
(this repo's dev machine has no CUDA GPU; bitsandbytes' 4-bit quantization
the trainers default to doesn't work without one, so the trainers'
run_*_training() bodies themselves aren't exercised here — see
tests/test_finetuning_sources.py's module docstring for the same note).
"""

from __future__ import annotations

import json

import pytest

from src.finetuning.manifest import split_by_repository, write_manifest


def _examples(*repo_and_count: tuple[str | None, int]) -> list[dict]:
    out = []
    for i, (repo, count) in enumerate(repo_and_count):
        out.extend({"task": f"t{i}-{j}", "solution": f"s{i}-{j}", "repository_url": repo} for j in range(count))
    return out


def test_single_repository_falls_back_to_per_example_split():
    examples = _examples(("https://git/x/y", 10))
    train, holdout = split_by_repository(examples, holdout_ratio=0.2, seed=1)
    assert len(train) + len(holdout) == 10
    assert len(holdout) == 2
    # every held-out example still belongs to the one repository
    assert all(e["repository_url"] == "https://git/x/y" for e in holdout)


def test_no_repository_field_is_treated_as_one_group():
    examples = [{"task": f"t{i}", "solution": f"s{i}"} for i in range(10)]  # no repository_url key at all
    train, holdout = split_by_repository(examples, holdout_ratio=0.3, seed=1)
    assert len(train) + len(holdout) == 10
    assert len(holdout) == 3


def test_multiple_repositories_never_split_a_single_repo_across_both_sides():
    examples = _examples(("repo-a", 5), ("repo-b", 5), ("repo-c", 5), ("repo-d", 5), ("repo-e", 5))
    train, holdout = split_by_repository(examples, holdout_ratio=0.2, seed=7)
    assert len(train) + len(holdout) == 25

    train_repos = {e["repository_url"] for e in train}
    holdout_repos = {e["repository_url"] for e in holdout}
    assert train_repos.isdisjoint(holdout_repos)
    assert len(holdout_repos) >= 1
    assert len(train_repos) >= 1


def test_two_repositories_always_leaves_at_least_one_in_train():
    examples = _examples(("repo-a", 3), ("repo-b", 3))
    train, _holdout = split_by_repository(examples, holdout_ratio=0.9, seed=3)  # even a very high ratio
    train_repos = {e["repository_url"] for e in train}
    assert len(train_repos) >= 1
    assert len(train) > 0


def test_split_is_deterministic_for_a_fixed_seed():
    examples = _examples(("repo-a", 4), ("repo-b", 4), ("repo-c", 4))
    first = split_by_repository(examples, seed=99)
    second = split_by_repository(examples, seed=99)
    assert first == second


def test_empty_input_produces_empty_splits():
    assert split_by_repository([]) == ([], [])


def test_write_manifest_produces_real_files_with_correct_hashes(tmp_path):
    train = [{"task": "t1", "solution": "s1", "repository_url": "https://git/x/y"}]
    holdout = [{"task": "t2", "solution": "s2", "repository_url": "https://git/a/b"}]

    manifest = write_manifest(train, holdout, str(tmp_path))

    train_path = tmp_path / "train.jsonl"
    holdout_path = tmp_path / "holdout.jsonl"
    manifest_path = tmp_path / "manifest.json"
    assert train_path.exists()
    assert holdout_path.exists()
    assert manifest_path.exists()

    # repository_url stripped from the written records (bookkeeping only)
    written_train = json.loads(train_path.read_text().strip())
    assert "repository_url" not in written_train
    assert written_train == {"task": "t1", "solution": "s1"}

    assert manifest["train_count"] == 1
    assert manifest["holdout_count"] == 1
    assert manifest["train_repositories"] == ["https://git/x/y"]
    assert manifest["holdout_repositories"] == ["https://git/a/b"]

    import hashlib

    assert manifest["train_sha256"] == hashlib.sha256(train_path.read_bytes()).hexdigest()
    assert manifest["holdout_sha256"] == hashlib.sha256(holdout_path.read_bytes()).hexdigest()

    # loaded back from disk it's the real manifest.json, not just the return value
    assert json.loads(manifest_path.read_text()) == manifest


def test_write_manifest_with_no_repository_field_reports_empty_repository_lists(tmp_path):
    train = [{"task": "t1", "solution": "s1"}]
    manifest = write_manifest(train, [], str(tmp_path))
    assert manifest["train_repositories"] == []
    assert manifest["holdout_repositories"] == []


@pytest.mark.parametrize("holdout_ratio", [0.1, 0.2, 0.3, 0.5])
def test_split_never_loses_or_duplicates_examples(holdout_ratio):
    examples = _examples(("r1", 7), ("r2", 3), ("r3", 12), ("r4", 1))
    train, holdout = split_by_repository(examples, holdout_ratio=holdout_ratio, seed=5)
    all_ids = {e["task"] for e in train} | {e["task"] for e in holdout}
    assert len(train) + len(holdout) == len(examples)
    assert all_ids == {e["task"] for e in examples}
