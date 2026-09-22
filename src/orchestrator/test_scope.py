# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — scoped test selection.

Maps the files the agent touched to the tests most likely to cover them,
so the testing node (and the agent's own `run_tests` tool) can run those
first — seconds of feedback on a large repo instead of minutes — and only
then the full suite. Pure functions over path sets; the commands they
return run inside the sandbox via `Workspace.run`.

What "related" means per ecosystem, honestly scoped:

- **python**: `tests/test_<stem>.py`, `test_<stem>.py` / `<stem>_test.py`
  next to the file or under a `tests/` dir at any depth, plus any touched
  file that is itself a test module. Run with `pytest <files>`.
- **node (jest)**: jest's own `--findRelatedTests <files>`, which walks the
  module graph — only when the repo's test script is jest.
- **go**: `go test` on the packages (directories) of the touched files.
- **rust**: none — `cargo test` is already per-crate and incremental.

`None` means "nothing narrower than the full suite is known"; callers then
run the full command directly rather than a scoped run that would collect
nothing.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from src.orchestrator.repo_profile import RepoProfile


def _is_python_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def python_related_tests(touched: list[str], tracked_files: set[str]) -> list[str]:
    """Test modules in `tracked_files` that plausibly cover `touched` (see module docstring)."""
    related: list[str] = []
    test_files = [f for f in tracked_files if _is_python_test_file(f) and PurePosixPath(f).name != "conftest.py"]
    for path in touched:
        if not path.endswith(".py"):
            continue
        if _is_python_test_file(path):
            # A touched test module is itself the most related test; conftest.py is shared fixtures, not tests.
            if path in tracked_files and path not in related and PurePosixPath(path).name != "conftest.py":
                related.append(path)
            continue
        stem = PurePosixPath(path).stem
        if stem == "__init__":
            continue
        wanted = {f"test_{stem}.py", f"{stem}_test.py"}
        for test_file in sorted(test_files):
            if PurePosixPath(test_file).name in wanted and test_file not in related:
                related.append(test_file)
    return related


def related_test_command(profile: RepoProfile, touched: list[str], tracked_files: set[str]) -> str | None:
    """The scoped command for this ecosystem, or None when only the full suite applies."""
    if not touched or not profile.test_cmd:
        return None
    if profile.ecosystem == "python":
        files = python_related_tests(touched, tracked_files)
        if not files:
            return None
        return "pytest -q " + " ".join(shlex.quote(f) for f in files)
    if profile.ecosystem == "node":
        if "jest" not in profile.test_cmd and "jest" not in profile.test_script:
            return None
        sources = [f for f in touched if f.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))]
        if not sources:
            return None
        return "npx jest --findRelatedTests --passWithNoTests " + " ".join(shlex.quote(f) for f in sources)
    if profile.ecosystem == "go":
        dirs = sorted({str(PurePosixPath(f).parent) for f in touched if f.endswith(".go")})
        if not dirs:
            return None
        return "go test " + " ".join(shlex.quote(f"./{d}" if d != "." else ".") for d in dirs)
    return None
