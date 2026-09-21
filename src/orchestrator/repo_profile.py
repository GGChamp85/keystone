# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — repo ecosystem detection.

Replaces `nodes/testing.py`'s old heuristic per-language command guessing
(py_compile / "pytest if a path contains 'test'" / bare `import`) with real
detection of what a repo's own tooling actually is, so the testing node runs
the repo's real test suite instead of a synthetic smoke check. Detection is
pure file-presence/content sniffing against an already-cloned working tree
(no execution) — callers pass in the list of files at the repo root (and a
couple of well-known subpaths) via `RepoWorkspace`-shaped read access; see
`src/orchestrator/workspace.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RepoProfile:
    ecosystem: str  # python | node | go | rust | unknown
    install_cmd: str | None
    test_cmd: str | None
    lint_cmds: list[str] = field(default_factory=list)
    typecheck_cmd: str | None = None


def _has(files: set[str], *names: str) -> bool:
    return any(n in files for n in names)


def detect_repo_profile(files: set[str], package_json_scripts: dict[str, str] | None = None) -> RepoProfile:
    """
    `files` is the set of relative paths present at (or near) the repo root
    — callers typically pass the output of a shallow `find . -maxdepth 2`
    inside the cloned sandbox workspace. `package_json_scripts` is the
    already-parsed `scripts` object from `package.json`, when present,
    since npm/yarn/pnpm test/lint commands are project-defined rather than
    inferable from file presence alone.
    """
    if _has(files, "pyproject.toml", "setup.py", "requirements.txt", "requirements-dev.txt", "Pipfile"):
        return _python_profile(files)
    if _has(files, "package.json"):
        return _node_profile(files, package_json_scripts or {})
    if _has(files, "go.mod"):
        return RepoProfile(
            ecosystem="go",
            install_cmd="go mod download",
            test_cmd="go test ./...",
            lint_cmds=["go vet ./..."],
        )
    if _has(files, "Cargo.toml"):
        return RepoProfile(
            ecosystem="rust",
            install_cmd="cargo fetch",
            test_cmd="cargo test",
            lint_cmds=["cargo clippy --all-targets -- -D warnings"],
        )
    return RepoProfile(ecosystem="unknown", install_cmd=None, test_cmd=None)


def _python_profile(files: set[str]) -> RepoProfile:
    if _has(files, "requirements-dev.txt"):
        install = (
            "pip install -r requirements.txt -r requirements-dev.txt"
            if _has(files, "requirements.txt")
            else "pip install -r requirements-dev.txt"
        )
    elif _has(files, "requirements.txt"):
        install = "pip install -r requirements.txt"
    elif _has(files, "pyproject.toml"):
        install = "pip install -e '.[dev]'" if _has(files, "poetry.lock") is False else "pip install -e ."
    else:
        install = "pip install -e ."

    test_cmd = (
        "pytest"
        if _has(files, "pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py")
        else "python -m unittest discover"
    )

    lint_cmds: list[str] = []
    typecheck_cmd: str | None = None
    if _has(files, "ruff.toml", ".ruff.toml", "pyproject.toml"):
        lint_cmds.append("ruff check . --output-format=json")
    if _has(files, ".flake8"):
        lint_cmds.append("flake8")
    if _has(files, "mypy.ini", "pyproject.toml", "setup.cfg"):
        typecheck_cmd = "mypy ."
    if _has(files, ".bandit", "pyproject.toml"):
        lint_cmds.append("bandit -q -r . -f json")

    return RepoProfile(
        ecosystem="python", install_cmd=install, test_cmd=test_cmd, lint_cmds=lint_cmds, typecheck_cmd=typecheck_cmd
    )


def _node_profile(files: set[str], scripts: dict[str, str]) -> RepoProfile:
    pm = "npm"
    install = "npm ci"
    if _has(files, "pnpm-lock.yaml"):
        pm, install = "pnpm", "pnpm install --frozen-lockfile"
    elif _has(files, "yarn.lock"):
        pm, install = "yarn", "yarn install --frozen-lockfile"

    test_cmd = f"{pm} test" if "test" in scripts else None
    lint_cmds = [f"{pm} run lint"] if "lint" in scripts else []
    if _has(files, ".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.cjs") and "lint" not in scripts:
        lint_cmds.append("npx eslint . -f json")
    typecheck_cmd = (
        f"{pm} run typecheck"
        if "typecheck" in scripts
        else ("npx tsc --noEmit" if _has(files, "tsconfig.json") else None)
    )

    return RepoProfile(
        ecosystem="node", install_cmd=install, test_cmd=test_cmd, lint_cmds=lint_cmds, typecheck_cmd=typecheck_cmd
    )
