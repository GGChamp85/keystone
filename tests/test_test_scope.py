# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""src/orchestrator/test_scope.py — which tests are "related" to the files the agent touched, per ecosystem."""

from __future__ import annotations

from src.orchestrator.repo_profile import RepoProfile, detect_repo_profile
from src.orchestrator.test_scope import python_related_tests, related_test_command

_PY_TREE = {
    "pyproject.toml",
    "src/pkg/__init__.py",
    "src/pkg/bucket.py",
    "src/pkg/io.py",
    "tests/conftest.py",
    "tests/test_bucket.py",
    "tests/unit/test_io.py",
    "src/pkg/bucket_test.py",
}


def test_python_related_tests_finds_tests_by_stem_next_to_and_under_tests():
    assert python_related_tests(["src/pkg/bucket.py"], _PY_TREE) == ["src/pkg/bucket_test.py", "tests/test_bucket.py"]
    assert python_related_tests(["src/pkg/io.py"], _PY_TREE) == ["tests/unit/test_io.py"]


def test_python_related_tests_includes_a_touched_test_module_and_skips_init_and_conftest():
    assert python_related_tests(["tests/test_bucket.py", "src/pkg/__init__.py"], _PY_TREE) == ["tests/test_bucket.py"]
    assert python_related_tests(["tests/conftest.py"], _PY_TREE) == []


def test_python_scoped_command_quotes_paths_and_returns_none_without_candidates():
    profile = detect_repo_profile({"pyproject.toml"})
    assert related_test_command(profile, ["src/pkg/bucket.py"], _PY_TREE) == (
        "pytest -q src/pkg/bucket_test.py tests/test_bucket.py"
    )
    assert related_test_command(profile, ["src/pkg/unrelated.py"], _PY_TREE) is None
    assert related_test_command(profile, [], _PY_TREE) is None


def test_node_uses_jest_find_related_tests_only_for_jest_projects():
    jest = detect_repo_profile({"package.json"}, {"test": "jest --coverage"})
    assert jest.test_script == "jest --coverage"
    assert related_test_command(jest, ["src/a.ts", "README.md"], set()) == (
        "npx jest --findRelatedTests --passWithNoTests src/a.ts"
    )
    mocha = detect_repo_profile({"package.json"}, {"test": "mocha"})
    assert related_test_command(mocha, ["src/a.ts"], set()) is None


def test_go_scopes_to_the_touched_packages():
    go = detect_repo_profile({"go.mod"})
    assert related_test_command(go, ["internal/auth/token.go", "cmd/main.go", "README.md"], set()) == (
        "go test ./cmd ./internal/auth"
    )


def test_rust_and_unknown_have_no_scoped_command():
    assert related_test_command(detect_repo_profile({"Cargo.toml"}), ["src/lib.rs"], set()) is None
    assert related_test_command(RepoProfile("unknown", None, None), ["x.py"], set()) is None
