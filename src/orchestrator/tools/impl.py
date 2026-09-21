# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — real tool implementations.

Every tool executes against a real `Workspace` (src/orchestrator/workspace.py
— a real sandboxed git checkout), never against in-memory state. Every
result is a `ToolResult`: `ok=False` is a real, expected outcome (a search
string not found, a patch that doesn't apply, a command that exits
non-zero) carrying an actionable `error` message the model can act on —
not an exception. A tool implementation only raises for something outside
the model's control (the sandbox itself unreachable); those are left to
propagate so the node they're called from decides how to handle
infrastructure failure, which is a different problem than "the model's
patch didn't apply."
"""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from src.orchestrator.workspace import Workspace

MAX_OUTPUT_CHARS = 20_000
MAX_TIMEOUT_SECONDS = 300
DEFAULT_COMMAND_TIMEOUT = 60


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""

    def to_content(self) -> str:
        """The `content` of a `role: tool` message — plain text, not JSON, since it's meant to be read, not parsed."""
        return self.output if self.ok else f"ERROR: {self.error}"


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


async def _read_file_checked(ws: Workspace, path: str) -> tuple[str | None, str | None]:
    """(content, error) — `error` set for a missing file (a normal, expected outcome the model should see
    plainly); any other transport failure (sandbox unreachable, 5xx) propagates as a real exception."""
    try:
        return await ws.read_file(path), None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None, f"{path!r} does not exist in the repository."
        raise


async def read_file(ws: Workspace, path: str) -> ToolResult:
    content, error = await _read_file_checked(ws, path)
    if error:
        return ToolResult(ok=False, error=error)
    return ToolResult(ok=True, output=_truncate(content or ""))


async def list_dir(ws: Workspace, path: str = ".", max_depth: int = 2) -> ToolResult:
    safe_path = shlex.quote(path)
    result = await ws.run(f"find {safe_path} -maxdepth {int(max_depth)} -not -path '*/.git*'", check=False)
    if result["exit_code"] != 0:
        return ToolResult(ok=False, error=result.get("stderr", "").strip() or f"list_dir failed for {path!r}")
    return ToolResult(ok=True, output=_truncate(result.get("stdout", "")))


async def grep(ws: Workspace, pattern: str, path: str = ".", case_insensitive: bool = False) -> ToolResult:
    flags = "-rn --exclude-dir=.git"
    if case_insensitive:
        flags += "i"
    safe_pattern = shlex.quote(pattern)
    safe_path = shlex.quote(path)
    # grep exits 1 for "no matches" — that's a real, useful answer, not a
    # tool failure (a model asking "does this pattern exist?" needs to see
    # "no" as cleanly as "yes").
    result = await ws.run(f"grep -E {flags} {safe_pattern} {safe_path}", check=False)
    if result["exit_code"] == 1:
        return ToolResult(ok=True, output="(no matches)")
    if result["exit_code"] not in (0, 1):
        return ToolResult(ok=False, error=result.get("stderr", "").strip() or "grep failed")
    return ToolResult(ok=True, output=_truncate(result.get("stdout", "")))


async def run_command(ws: Workspace, command: str, timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT) -> ToolResult:
    timeout = min(max(1, timeout_seconds), MAX_TIMEOUT_SECONDS)
    # check=False: a non-zero exit is a normal, useful tool result (below),
    # not a Python exception. A genuinely unreachable sandbox raises
    # httpx.* here and is left to propagate — that's an infra failure for
    # the calling node to handle, not something a ToolResult should hide.
    result = await ws.run(command, timeout=timeout, check=False)
    exit_code = result.get("exit_code", -1)
    stdout = _truncate(result.get("stdout", ""))
    stderr = _truncate(result.get("stderr", ""))
    output = f"$ {command}\n(exit code {exit_code})\nstdout:\n{stdout}"
    if stderr.strip():
        output += f"\nstderr:\n{stderr}"
    # A non-zero exit is real, useful information for the model (e.g. "the
    # command it asked to run failed") — surfaced as ok=True with the exit
    # code visible in the output, not as a tool error; ok=False here is
    # reserved for the tool call itself being malformed or the sandbox
    # being unreachable.
    return ToolResult(ok=True, output=output)


async def apply_patch(
    ws: Workspace,
    *,
    path: str | None = None,
    search: str | None = None,
    replace: str | None = None,
    create: bool = False,
    delete: bool = False,
    unified_diff: str | None = None,
) -> ToolResult:
    if unified_diff:
        return await _apply_unified_diff(ws, unified_diff)
    if delete:
        if not path:
            return ToolResult(ok=False, error="apply_patch: delete=true requires `path`")
        result = await ws.run(f"rm -f -- {shlex.quote(path)}", check=False)
        if result["exit_code"] != 0:
            return ToolResult(ok=False, error=result.get("stderr", "").strip() or f"Could not delete {path!r}")
        return ToolResult(ok=True, output=f"Deleted {path}")
    if create:
        if not path:
            return ToolResult(ok=False, error="apply_patch: create=true requires `path`")
        existing = await ws.run(f"test -f {shlex.quote(path)}", check=False)
        if existing["exit_code"] == 0:
            return ToolResult(
                ok=False,
                error=f"{path!r} already exists — use search/replace to edit it, not create=true "
                "(which is for new files only)",
            )
        await ws.write_file(path, replace or "")
        return ToolResult(ok=True, output=f"Created {path} ({len(replace or '')} bytes)")
    if search is not None:
        return await _apply_search_replace(ws, path, search, replace or "")
    return ToolResult(
        ok=False,
        error="apply_patch: give either (path, search, replace), (path, create=true, replace), "
        "(path, delete=true), or unified_diff",
    )


async def _apply_search_replace(ws: Workspace, path: str | None, search: str, replace: str) -> ToolResult:
    if not path:
        return ToolResult(ok=False, error="apply_patch: search/replace mode requires `path`")
    content, error = await _read_file_checked(ws, path)
    if error or content is None:
        return ToolResult(ok=False, error=f"{path!r} does not exist — use create=true for a new file")

    count = content.count(search)
    if count == 0:
        return ToolResult(
            ok=False,
            error=f"The `search` text was not found in {path!r}. Re-read the file — it may have changed "
            "since you last saw it — and copy the exact text (including whitespace/indentation) to match.",
        )
    if count > 1:
        return ToolResult(
            ok=False,
            error=f"The `search` text matches {count} places in {path!r}, ambiguous. Include more "
            "surrounding context in `search` so it matches exactly once.",
        )

    new_content = content.replace(search, replace, 1)
    await ws.write_file(path, new_content)
    return ToolResult(ok=True, output=f"Updated {path} ({len(new_content)} bytes)")


async def _apply_unified_diff(ws: Workspace, diff_text: str) -> ToolResult:
    patch_file = "_keystone_patch.diff"
    await ws.write_file(patch_file, diff_text if diff_text.endswith("\n") else diff_text + "\n")
    result = await ws.run(f"git apply --3way {shlex.quote(patch_file)}", check=False)
    await ws.run(f"rm -f -- {shlex.quote(patch_file)}", check=False)
    if result["exit_code"] != 0:
        return ToolResult(
            ok=False,
            error="git apply failed:\n" + _truncate(result.get("stderr", "") or result.get("stdout", ""), 4000),
        )
    return ToolResult(ok=True, output="Patch applied.\n" + _truncate(result.get("stdout", "")))


TOOL_IMPLS: dict[str, Callable[..., Awaitable[ToolResult]]] = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "apply_patch": apply_patch,
    "run_command": run_command,
}


async def dispatch_tool_call(ws: Workspace, name: str, arguments: dict[str, Any]) -> ToolResult:
    """
    Look up and invoke a tool by name with the model-supplied `arguments`,
    for the agentic loop (`nodes/coding.py`) to call without knowing each
    tool's exact signature. A model hallucinating a tool name or passing
    arguments that don't match the schema (extra/misspelled keys, wrong
    types) is a normal, expected failure mode here — turned into a
    `ToolResult`, same as any other tool-level failure, not an exception
    that would end the whole loop over one bad call.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return ToolResult(ok=False, error=f"Unknown tool {name!r}. Available tools: {sorted(TOOL_IMPLS)}")
    try:
        return await impl(ws, **arguments)
    except TypeError as exc:
        return ToolResult(ok=False, error=f"Invalid arguments for {name!r}: {exc}")


def touched_path(name: str, arguments: dict[str, Any], result: ToolResult) -> str | None:
    """The file path a successful apply_patch call changed, or None — for the loop's `files_touched` bookkeeping."""
    if name != "apply_patch" or not result.ok:
        return None
    return arguments.get("path")
