# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — fine-tuning data source: a repository's real commit
history.

Operates on a real *local* git checkout (a `git.Repo` GitPython handle) —
not a URL it clones itself. A caller with a real reachable git host
(src/git/gitea.py, src/memory/ingestion.py's SSRF-validated clone) hands
this an already-checked-out path; this module never talks to a network
itself, so it's testable against a real local repo with no external git
host required (the same gap tests/test_git_workflow_integration.py
already documents for a real *reachable* one).

Each qualifying commit becomes one real instruction -> diff SFT example:
`instruction` is the commit's own real message, `solution` is its real
diff against its first parent. Filtered for the reasons the plan calls
for: too short a message to be a real instruction, a diff too large to be
a focused single-purpose example, a diff touching only generated/vendored
paths (a lockfile bump teaches nothing about writing code), and — via
src/sandbox/security.py's `scan_sandbox_output`, the same scanner the
sandbox runtime already uses on real command output — a diff that looks
like it's exfiltrating something isn't safe to feed into training data
either, whatever its message says.
"""

from __future__ import annotations

from src.sandbox.security import scan_sandbox_output

_GENERATED_PATH_MARKERS = (
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "Cargo.lock",
    "go.sum",
    ".min.js",
    ".min.css",
    "/dist/",
    "/build/",
    "/node_modules/",
    "/__pycache__/",
    "/vendor/",
    ".generated.",
)


def _looks_generated(paths: list[str]) -> bool:
    """A commit that ONLY touches generated/vendored/lockfile paths has no
    real hand-written instruction->code signal to teach from — a mixed
    commit (some real source, some lockfile) still counts as real, since
    the diff still contains genuine authored changes."""
    if not paths:
        return False
    return all(any(marker in p for marker in _GENERATED_PATH_MARKERS) for p in paths)


def build_sft_examples_from_git_history(
    repo_path: str,
    branch: str = "HEAD",
    max_commits: int = 500,
    min_message_length: int = 15,
    max_diff_size_kb: int = 200,
    repository_url: str | None = None,
) -> list[dict]:
    """
    Real commit history -> real SFT examples, from a local checkout at
    `repo_path`. `repository_url`, if given, is carried through on each
    example for src/finetuning/manifest.py's repository-stratified split —
    same convention as src/finetuning/sources/trajectories.py.
    """
    import git

    repo = git.Repo(repo_path)
    examples: list[dict] = []

    for commit in repo.iter_commits(branch, max_count=max_commits):
        if not commit.parents:
            continue  # the repo's very first commit has no parent to diff against

        message = commit.message.strip()
        if len(message) < min_message_length:
            continue

        parent = commit.parents[0]
        diff_text = repo.git.diff(parent.hexsha, commit.hexsha)
        if not diff_text.strip():
            continue
        if len(diff_text.encode()) > max_diff_size_kb * 1024:
            continue

        changed_paths = [d.b_path or d.a_path for d in commit.diff(parent) if (d.b_path or d.a_path)]
        if _looks_generated(changed_paths):
            continue

        scan = scan_sandbox_output(stdout="", stderr="", code=diff_text)
        if not scan.is_safe:
            continue

        examples.append(
            {
                "task": message,
                "context": f"Repository: {repository_url}" if repository_url else "",
                "solution": diff_text,
                "repository_url": repository_url,
            }
        )

    return examples
