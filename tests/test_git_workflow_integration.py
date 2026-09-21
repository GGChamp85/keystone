# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real end-to-end integration test for the Phase 1 git workflow: a sandboxed
`Workspace` clones a real repo from a real Gitea instance, runs the repo's
real test suite (via `repo_profile` detection, not a heuristic guess),
makes a real change, commits, pushes, and opens a real pull request via
`GiteaHost` — verified end-to-end during development against a live Gitea
1.22 container, a live sandbox daemon, and a live Docker sandbox (not
mocked at any layer).

Requires (see docs/deployment/... for the real setup, or bring up a throwaway
Gitea container the same way: `docker run -d --name gitea -p 3000:3000
gitea/gitea:1.22`, create an admin user via `gitea admin user create`, mint
a token via `POST /api/v1/users/{user}/tokens`):
  - SANDBOX_DAEMON_URL pointed at a running `keystoned` with a real Docker
    backend and the `keystone-sandbox-python:latest` image built
    (`make sandbox-images`).
  - GIT_HOST_API_URL / GIT_HOST_TOKEN pointed at a real Gitea instance
    reachable from wherever this test process runs.
  - GITEA_TEST_REPO_URL: the https:// clone URL of a repo the token can
    push to and open PRs on — must also be reachable via that same URL
    *from inside the sandbox* (i.e. on settings.git_allowed_hosts and on
    a network the sandbox daemon's containers can actually route to; the
    Gitea host and the sandbox's `keystone-sandbox-net` Docker network
    need to be the same, or bridged, for this to resolve).
"""

from __future__ import annotations

import os
import uuid

import httpx
import pytest

pytestmark = pytest.mark.integration


def _git_workflow_env_ready() -> bool:
    return bool(
        os.environ.get("SANDBOX_DAEMON_URL")
        and os.environ.get("GIT_HOST_TOKEN")
        and os.environ.get("GIT_HOST_API_URL")
        and os.environ.get("GITEA_TEST_REPO_URL")
    )


requires_git_workflow_env = pytest.mark.skipif(
    not _git_workflow_env_ready(),
    reason="Needs SANDBOX_DAEMON_URL, GIT_HOST_TOKEN, GIT_HOST_API_URL, GITEA_TEST_REPO_URL — see module docstring",
)


@pytest.fixture
async def seeded_repo():
    """
    Seeds GITEA_TEST_REPO_URL with a minimal, real, pytest-runnable Python
    project via Gitea's contents API (no local git needed) — a fresh commit
    per test run so this is repeatable, not dependent on manually-seeded
    state.
    """
    repo_url = os.environ["GITEA_TEST_REPO_URL"]
    token = os.environ["GIT_HOST_TOKEN"]
    api_url = os.environ["GIT_HOST_API_URL"].rstrip("/")

    from src.git.host import parse_owner_repo

    owner, repo = parse_owner_repo(repo_url)

    pyproject_toml = (
        '[project]\nname = "sample-repo"\nversion = "0.1.0"\n\n[tool.pytest.ini_options]\npythonpath = ["."]\n'
    )
    async with httpx.AsyncClient(base_url=api_url, headers={"Authorization": f"token {token}"}, timeout=30.0) as client:
        files = {
            "pyproject.toml": pyproject_toml,
            "app.py": "def add(a, b):\n    return a + b\n",
            "tests/test_app.py": "from app import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        }
        for path, content in files.items():
            import base64

            resp = await client.get(f"/repos/{owner}/{repo}/contents/{path}")
            payload = {
                "content": base64.b64encode(content.encode()).decode(),
                "message": f"test setup: seed {path}",
                "branch": "main",
            }
            if resp.status_code == 200:
                payload["sha"] = resp.json()["sha"]
                await client.put(f"/repos/{owner}/{repo}/contents/{path}", json=payload)
            else:
                await client.post(f"/repos/{owner}/{repo}/contents/{path}", json=payload)

    return repo_url


@requires_git_workflow_env
async def test_full_git_workflow_clone_test_branch_commit_push_pr(seeded_repo):
    from src.git.gitea import GiteaHost
    from src.git.host import parse_owner_repo
    from src.orchestrator.repo_profile import detect_repo_profile
    from src.orchestrator.workspace import Workspace
    from src.sandbox.manager import SandboxManager

    task_id = f"test-{uuid.uuid4().hex[:8]}"
    manager = SandboxManager()
    ws = Workspace(manager, task_id=task_id, tenant_id="test-tenant")

    try:
        result = await ws.clone(seeded_repo, branch="main")
        assert result.commit_sha

        files = await ws.list_files()
        profile = detect_repo_profile(files)
        assert profile.ecosystem == "python"
        assert profile.test_cmd == "pytest"

        baseline = await ws.run(profile.test_cmd, check=False)
        assert baseline["exit_code"] == 0, baseline["stdout"]

        branch_name = f"keystone/test-user/{task_id}"
        await ws.create_branch(branch_name)
        await ws.write_file("app.py", "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n")
        assert "subtract" in await ws.read_file("app.py")

        diff_stat = await ws.diff_stat()
        assert "app.py" in diff_stat

        sha = await ws.commit_all("Add a subtract function")
        assert sha and sha != result.commit_sha

        await ws.push(branch_name)

        owner, repo = parse_owner_repo(seeded_repo)
        host = GiteaHost()
        try:
            default_branch = await host.get_default_branch(owner, repo)
            pr = await host.open_pull_request(
                owner,
                repo,
                head=branch_name,
                base=default_branch,
                title="Add subtract function",
                body="Opened by tests/test_git_workflow_integration.py",
            )
            assert pr.number > 0
            assert pr.html_url

            status = await host.get_pull_request_status(owner, repo, pr.number)
            assert status.state == "open"
            assert not status.merged
        finally:
            await host.aclose()
    finally:
        await ws.close()
        await manager.aclose()
