#!/usr/bin/env python3
# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Truth-in-docs gate: every claim in docs/claims.yaml must (a) appear verbatim in the documents it is
attributed to and (b) name tests that really exist in this repository's collected test suite.

    python scripts/check_claims.py                # exit 1 on any broken claim
    python scripts/check_claims.py --collected f  # reuse a saved `pytest --collect-only -q` output

A claim without a test is not a claim this project makes; a test that no longer exists means the
sentence in the docs must change. CI runs this next to the docs build.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "docs" / "claims.yaml"


def collected_test_ids(saved: Path | None = None) -> set[str]:
    if saved is not None:
        text = saved.read_text()
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        text = proc.stdout
    ids = {line.strip() for line in text.splitlines() if "::" in line and not line.startswith(" ")}
    return ids


def test_exists(test_id: str, ids: set[str]) -> bool:
    if test_id in ids:
        return True
    if "::" not in test_id:  # a whole file: any collected test inside it
        return any(i.startswith(test_id + "::") for i in ids)
    # a parametrised test: "file::name" matches "file::name[param]"
    return any(i.startswith(test_id + "[") for i in ids)


def check(claims_path: Path = CLAIMS, saved_collection: Path | None = None) -> list[str]:
    data = yaml.safe_load(claims_path.read_text()) or {}
    ids = collected_test_ids(saved_collection)
    problems: list[str] = []
    for n, claim in enumerate(data.get("claims") or [], start=1):
        text = claim.get("text", "")
        docs = claim.get("docs") or []
        tests = claim.get("tests") or []
        label = f"claim {n} ({text[:60]!r})"
        if not text or not docs or not tests:
            problems.append(f"{label}: needs text, docs and tests")
            continue
        for doc in docs:
            path = ROOT / doc
            if not path.exists():
                problems.append(f"{label}: document {doc} does not exist")
            elif text not in path.read_text():
                problems.append(f"{label}: text not found verbatim in {doc}")
        problems.extend(f"{label}: test {t} is not collected by pytest" for t in tests if not test_exists(t, ids))
    return problems


def main(argv: list[str]) -> int:
    saved = Path(argv[argv.index("--collected") + 1]) if "--collected" in argv else None
    problems = check(saved_collection=saved)
    for p in problems:
        print(f"CLAIM BROKEN: {p}", file=sys.stderr)
    if problems:
        return 1
    n = len((yaml.safe_load(CLAIMS.read_text()) or {}).get("claims") or [])
    print(f"{n} documented claims verified against the collected test suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
