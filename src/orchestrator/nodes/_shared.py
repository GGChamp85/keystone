# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — shared helpers used by more than one graph node.

`get_or_clone_workspace` is the single place a repo-backed task's sandbox
gets created and the repository cloned — both `nodes/coding.py` (the
agentic tool loop needs the working tree before it can read/edit anything)
and `nodes/testing.py` (runs the repo's real test command against that same
working tree) call it, so there is exactly one clone path and exactly one
place `state.repo_cloned`/`state.working_branch` get set, no matter which
node happens to run first for a given task.
"""

from __future__ import annotations

import re

import structlog

from src.orchestrator.state import AgentPhase, AgentState
from src.orchestrator.workspace import Workspace
from src.sandbox.manager import get_sandbox_manager

logger = structlog.get_logger(__name__)


def slugify_for_branch(text: str) -> str:
    """Git-ref-safe slug (lowercase, alnum and '-' only, collapsed) for the
    `keystone/<slug>/<task_id>` working branch name — e.g. a user's email
    local-part ("ada.lovelace@x.com" -> "ada-lovelace")."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "user"


def next_gate_phase(state: AgentState) -> AgentPhase:
    """
    Where to go after a gate (coding's own completion, or quality) passes
    — REVIEW if enabled, else TESTING if enabled, else straight to
    COMPLETE. One place for this cascade so coding.py and quality.py don't
    each re-implement (and risk diverging on) the same enable-flag logic.
    """
    if state.enable_reasoning_review:
        return AgentPhase.REVIEW
    if state.enable_sandbox_testing:
        return AgentPhase.TESTING
    return AgentPhase.COMPLETE


async def get_or_clone_workspace(state: AgentState) -> Workspace:
    """
    Returns a `Workspace` over the task's shared sandbox (created once,
    reused for the task's lifetime via `get_sandbox_manager()`'s singleton
    handle cache). Clones `state.repository_url` and creates the agent's
    working branch on first use; a later call for the same task_id reuses
    the already-cloned working tree untouched, so a fixing/testing
    iteration never re-clones over in-progress edits.
    """
    manager = get_sandbox_manager()
    ws = Workspace(manager, task_id=str(state.task_id), tenant_id=str(state.tenant_id or "default"))
    await ws.ensure_sandbox(network_enabled=True)

    if not state.repo_cloned:
        if not state.repository_url:
            raise ValueError("get_or_clone_workspace requires state.repository_url")
        clone_result = await ws.clone(state.repository_url, state.branch)
        state.working_branch = f"keystone/{state.user_slug or 'agent'}/{state.task_id}"
        await ws.create_branch(state.working_branch)
        state.repo_cloned = True
        logger.info(
            "workspace.repo_ready",
            task_id=str(state.task_id),
            repo=state.repository_url,
            base_sha=clone_result.commit_sha[:12],
            working_branch=state.working_branch,
        )

    return ws
