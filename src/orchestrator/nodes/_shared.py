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

from src.config import Settings, get_settings
from src.orchestrator.repo_map import build_repo_map
from src.orchestrator.repo_profile import RepoProfile, detect_repo_profile
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
        await install_dependencies(state, ws)

    return ws


async def detect_profile(ws: Workspace) -> RepoProfile:
    """The repo's real tooling (src/orchestrator/repo_profile.py) from the live working tree."""
    files = await ws.list_files()
    scripts = await ws.read_package_json_scripts() if "package.json" in files else {}
    return detect_repo_profile(files, scripts)


def _install_env(settings: Settings) -> dict[str, str]:
    # Fail fast, not after minutes of retries, when the sandbox's egress policy
    # blocks the public index; point at the configured internal mirrors when set.
    env = {
        "PIP_RETRIES": "1",
        "PIP_DEFAULT_TIMEOUT": "20",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INPUT": "1",
        "npm_config_fetch_retries": "1",
        "CI": "true",
    }
    if settings.pip_index_url:
        env["PIP_INDEX_URL"] = settings.pip_index_url
    if settings.npm_registry_url:
        env["NPM_CONFIG_REGISTRY"] = settings.npm_registry_url
    if settings.go_proxy_url:
        env["GOPROXY"] = settings.go_proxy_url
    return env


async def install_dependencies(state: AgentState, ws: Workspace) -> None:
    """
    Run the repo's own install command once per task so its real test suite
    can import its real dependencies. Never raises and never fails the task:
    an install that cannot complete (no package mirror reachable from the
    sandbox, a broken lockfile) is recorded in `state.deps_install_error`
    and shown to the coding model, which is what turns "ModuleNotFoundError"
    from a mystery into an attributed cause.
    """
    if state.deps_installed:
        return
    state.deps_installed = True
    try:
        profile = await detect_profile(ws)
    except Exception as exc:
        state.deps_install_error = f"could not detect the repo's tooling: {exc}"
        logger.warning("deps.profile_failed", task_id=str(state.task_id), error=str(exc))
        return
    if not profile.install_cmd:
        return
    try:
        settings = get_settings()
        result = await ws.run(
            profile.install_cmd,
            timeout=settings.agent_install_timeout_seconds,
            env_vars=_install_env(settings),
            check=False,
        )
    except Exception as exc:
        state.deps_install_error = f"`{profile.install_cmd}` could not run: {exc}"
        logger.warning("deps.install_errored", task_id=str(state.task_id), error=str(exc))
        return
    if result.get("exit_code") != 0:
        output = (result.get("stderr") or result.get("stdout") or "").strip()
        state.deps_install_error = f"`{profile.install_cmd}` exited {result.get('exit_code')}:\n{output}"
        logger.warning("deps.install_failed", task_id=str(state.task_id), command=profile.install_cmd)
    else:
        logger.info(
            "deps.installed",
            task_id=str(state.task_id),
            command=profile.install_cmd,
            duration_ms=result.get("duration_ms"),
        )


async def ensure_repo_map(state: AgentState, ws: Workspace) -> str:
    """
    Build the repository map (src/orchestrator/repo_map.py) once per task and
    cache it on the state, so planning and coding share one map. Best-effort:
    a map that cannot be built (ctags crashed, sandbox hiccup) is logged and
    the task proceeds without it — the tools still work, the model just
    starts with less context. Never raises.
    """
    if state.repo_map or not state.repository_url:
        return state.repo_map
    try:
        state.repo_map = await build_repo_map(ws)
        logger.info("repo_map.built", task_id=str(state.task_id), chars=len(state.repo_map))
    except Exception as exc:  # any failure here must not fail the task
        logger.warning("repo_map.build_failed", task_id=str(state.task_id), error=str(exc))
    return state.repo_map
