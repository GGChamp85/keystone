# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — fine-tuning train/holdout split + manifest.

Splits a data source's examples (src/finetuning/sources/) into train and
holdout sets *stratified by repository* — a whole repository's examples
always land entirely on one side, never split across both. Without that,
a holdout "eval" number would be partly measuring memorization of that
exact repo's own style rather than real generalization, since the same
repo's conventions would already be in the training set. Records exactly
what was trained on (`manifest.json`, real content hashes of the written
files) — shown to a user before a run starts, and kept so a later question
("what data produced this adapter?") has a real, reproducible answer.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

_NO_REPOSITORY_KEY = "__no_repository__"


def split_by_repository(
    examples: list[dict],
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Groups `examples` by their `"repository_url"` key (missing/None/""
    collapse into one shared group) and assigns whole groups to the
    holdout split via a seeded shuffle, so no repository's examples appear
    on both sides.

    Falls back to a plain per-example seeded split when there's only one
    group (or none) — repo-stratification is meaningless with a single
    repository, but a holdout set still matters, so one is still carved
    out per-example rather than skipped.
    """
    if not examples:
        return [], []

    groups: dict[str, list[dict]] = {}
    for ex in examples:
        key = ex.get("repository_url") or _NO_REPOSITORY_KEY
        groups.setdefault(key, []).append(ex)

    rng = random.Random(seed)  # noqa: S311 — a deterministic dataset split, not a security control

    if len(groups) <= 1:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        cut = max(1, round(len(shuffled) * holdout_ratio)) if len(shuffled) > 1 else 0
        return shuffled[cut:], shuffled[:cut]

    repo_keys = list(groups.keys())
    rng.shuffle(repo_keys)
    holdout_count = max(1, round(len(repo_keys) * holdout_ratio))
    holdout_count = min(holdout_count, len(repo_keys) - 1)  # always leave at least one repo for train
    holdout_keys = set(repo_keys[:holdout_count])

    train: list[dict] = []
    holdout: list[dict] = []
    for key, group in groups.items():
        (holdout if key in holdout_keys else train).extend(group)
    return train, holdout


def _write_jsonl(rows: list[dict], path: Path) -> str:
    # repository_url is split-time bookkeeping (which group a row belongs
    # to), not part of any data_prep.py record shape — stripped on write so
    # the file is exactly what a trainer expects to load.
    clean = [{k: v for k, v in r.items() if k != "repository_url"} for r in rows]
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in clean)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repositories(rows: list[dict]) -> list[str]:
    return sorted({r["repository_url"] for r in rows if r.get("repository_url")})


def write_manifest(train: list[dict], holdout: list[dict], output_dir: str) -> dict:
    """Writes train.jsonl/holdout.jsonl plus manifest.json — real sha256
    content hashes of what was actually written (not the in-memory examples,
    the files a trainer will actually read), and each split's repository
    coverage, so a promotion decision or a later audit has real provenance
    instead of "some fine-tuning happened at some point"."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.jsonl"
    holdout_path = out / "holdout.jsonl"
    train_hash = _write_jsonl(train, train_path)
    holdout_hash = _write_jsonl(holdout, holdout_path)

    manifest = {
        "train_count": len(train),
        "holdout_count": len(holdout),
        "train_file": str(train_path),
        "train_sha256": train_hash,
        "holdout_file": str(holdout_path),
        "holdout_sha256": holdout_hash,
        "train_repositories": _repositories(train),
        "holdout_repositories": _repositories(holdout),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
