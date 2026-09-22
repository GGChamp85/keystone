# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""Unit tests for src/orchestrator/repo_profile.py — pure logic, no infra needed."""

from __future__ import annotations

from src.orchestrator.repo_profile import detect_repo_profile


def test_python_repo_with_pyproject_and_ruff():
    profile = detect_repo_profile({"pyproject.toml", "src", "tests"})
    assert profile.ecosystem == "python"
    assert profile.test_cmd == "pytest"
    assert "ruff check . --output-format=json" in profile.lint_cmds


def test_python_repo_with_requirements_txt_only():
    profile = detect_repo_profile({"requirements.txt", "setup.cfg"})
    assert profile.ecosystem == "python"
    assert profile.install_cmd == "pip install -r requirements.txt"
    assert profile.test_cmd == "pytest"  # setup.cfg present -> pytest-configured


def test_node_repo_with_npm_scripts():
    profile = detect_repo_profile({"package.json", "tsconfig.json"}, {"test": "jest", "lint": "eslint ."})
    assert profile.ecosystem == "node"
    assert profile.test_cmd == "npm test"
    assert profile.lint_cmds == ["npm run lint"]
    assert profile.typecheck_cmd == "npx tsc --noEmit"


def test_node_repo_with_pnpm_lockfile():
    profile = detect_repo_profile({"package.json", "pnpm-lock.yaml"}, {"test": "vitest"})
    assert profile.ecosystem == "node"
    assert profile.install_cmd == "pnpm install --frozen-lockfile"
    assert profile.test_cmd == "pnpm test"


def test_go_repo():
    profile = detect_repo_profile({"go.mod", "main.go"})
    assert profile.ecosystem == "go"
    assert profile.test_cmd == "go test ./..."
    assert profile.lint_cmds == ["go vet ./..."]


def test_rust_repo():
    profile = detect_repo_profile({"Cargo.toml", "src"})
    assert profile.ecosystem == "rust"
    assert profile.test_cmd == "cargo test"


def test_unknown_repo_has_no_commands():
    profile = detect_repo_profile({"README.md", "LICENSE"})
    assert profile.ecosystem == "unknown"
    assert profile.install_cmd is None
    assert profile.test_cmd is None


def test_python_takes_priority_over_node_if_both_present():
    # a repo with both a Python backend and a Node frontend at root level —
    # python wins because pyproject.toml is checked first; this is a
    # documented limitation, not a silent surprise, so pin it with a test.
    profile = detect_repo_profile({"pyproject.toml", "package.json"})
    assert profile.ecosystem == "python"


def test_python_repo_recognised_by_test_config_alone_has_tests_but_nothing_to_install():
    # A pytest.ini / tox.ini / conftest.py repo with no packaging files is still a Python repo
    # (found for real: `run_tests` reported ecosystem='unknown' on exactly this layout).
    for marker in ("pytest.ini", "tox.ini", "conftest.py"):
        profile = detect_repo_profile({marker, "app.py"})
        assert profile.ecosystem == "python", marker
        assert profile.test_cmd == "pytest"
        assert profile.install_cmd is None
