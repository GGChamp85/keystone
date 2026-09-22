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

import difflib
import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import get_settings
from src.orchestrator.context import fit_to_tokens
from src.orchestrator.repo_map import DEFAULT_MAP_TOKENS, build_repo_map
from src.orchestrator.repo_profile import detect_repo_profile
from src.orchestrator.test_scope import related_test_command
from src.orchestrator.workspace import Workspace

DEFAULT_COMMAND_TIMEOUT = 60


def _output_budget_tokens() -> int:
    """How much of one tool result may enter the prompt: a third of the loop's context budget, so a single
    result can never crowd out the task and the conversation. Derived from the configured context, not a
    fixed number; the full text always remains in the sandbox (files) or the task record (outputs)."""
    return max(1_000, get_settings().agent_max_context_tokens // 3)


# A fuzzy search/replace match must be at least this similar to the file's text (difflib ratio)
# AND be the only window that clears the bar — a near-miss that could mean two places is refused.
FUZZY_MATCH_THRESHOLD = 0.92


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""

    def to_content(self) -> str:
        """The `content` of a `role: tool` message — plain text, not JSON, since it's meant to be read, not parsed."""
        return self.output if self.ok else f"ERROR: {self.error}"


def _truncate(text: str, what: str = "output") -> str:
    return fit_to_tokens(text, _output_budget_tokens(), keep="head", what=what)


async def _read_file_checked(ws: Workspace, path: str) -> tuple[str | None, str | None]:
    """(content, error) — `error` set for a missing file (a normal, expected outcome the model should see
    plainly); any other transport failure (sandbox unreachable, 5xx) propagates as a real exception."""
    try:
        return await ws.read_file(path), None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return None, f"{path!r} does not exist in the repository."
        raise


async def read_file(ws: Workspace, path: str, start_line: int | None = None, end_line: int | None = None) -> ToolResult:
    """Whole file, or an inclusive 1-based line range rendered with line numbers so the model can
    quote exact lines back (in apply_patch, or in a review comment) without counting by hand."""
    content, error = await _read_file_checked(ws, path)
    if error:
        return ToolResult(ok=False, error=error)
    text = content or ""
    if start_line is None and end_line is None:
        return ToolResult(ok=True, output=_truncate(text))
    lines = text.splitlines()
    first = max(1, int(start_line or 1))
    last = min(len(lines), int(end_line or len(lines)))
    if first > len(lines):
        return ToolResult(ok=False, error=f"{path!r} has only {len(lines)} lines; start_line {first} is past the end.")
    if last < first:
        return ToolResult(ok=False, error=f"end_line ({last}) is before start_line ({first}).")
    width = len(str(last))
    numbered = "\n".join(f"{n:>{width}}: {lines[n - 1]}" for n in range(first, last + 1))
    header = f"{path} lines {first}-{last} of {len(lines)}"
    return ToolResult(ok=True, output=_truncate(f"{header}\n{numbered}"))


async def get_repo_map(ws: Workspace, max_tokens: int = DEFAULT_MAP_TOKENS) -> ToolResult:
    """The ranked symbol outline from src/orchestrator/repo_map.py, on demand and at the model's chosen size."""
    budget = min(max(500, int(max_tokens)), 16_000)
    return ToolResult(ok=True, output=await build_repo_map(ws, max_tokens=budget))


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
    timeout = min(max(1, timeout_seconds), get_settings().agent_tool_timeout_seconds)
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


def _tail(text: str, what: str = "test output") -> str:
    """Keep the END of long output — test runners put the failures and the summary line last."""
    return fit_to_tokens(text, _output_budget_tokens(), keep="tail", what=what)


async def run_tests(ws: Workspace, scope: str = "related", touched_paths: list[str] | None = None) -> ToolResult:
    """
    Run the repo's own test suite (detected by src/orchestrator/repo_profile.py,
    the same detection the testing node uses) — `scope="related"` narrows it to
    the tests covering the files this task has changed (src/orchestrator/test_scope.py),
    `scope="all"` runs everything. The exit code and the tail of the output come
    back as a normal result: a failing test is information for the model, not a
    tool error.
    """
    if scope not in ("related", "all"):
        return ToolResult(ok=False, error="run_tests: scope must be 'related' or 'all'")
    files = await ws.list_files()
    scripts = await ws.read_package_json_scripts() if "package.json" in files else {}
    profile = detect_repo_profile(files, scripts)
    if not profile.test_cmd:
        return ToolResult(
            ok=False,
            error=f"No test command could be detected for this repository (ecosystem={profile.ecosystem!r}). "
            "If it has one, run it explicitly with run_command.",
        )
    command = profile.test_cmd
    label = "full suite"
    note = ""
    if scope == "related":
        scoped = related_test_command(profile, touched_paths or [], await ws.list_tracked_files())
        if scoped is None:
            note = "\n(no tests related to the files changed so far were found — ran the full suite)"
        else:
            command, label = scoped, "related tests"
    result = await ws.run(command, timeout=get_settings().agent_test_timeout_seconds, check=False)
    code = result.get("exit_code", -1)
    verdict = "passed" if code == 0 else ("collected no tests" if code == 5 else "FAILED")
    body = _tail((result.get("stdout", "") + "\n" + result.get("stderr", "")).strip())
    return ToolResult(ok=True, output=f"$ {command}\n(exit code {code}) {label} {verdict}{note}\n{body}")


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
        return await _apply_fuzzy_search_replace(ws, path, content, search, replace)
    if count > 1:
        return ToolResult(
            ok=False,
            error=f"The `search` text matches {count} places in {path!r}, ambiguous. Include more "
            "surrounding context in `search` so it matches exactly once.",
        )

    new_content = content.replace(search, replace, 1)
    await ws.write_file(path, new_content)
    return ToolResult(ok=True, output=f"Updated {path} ({len(new_content)} bytes)")


@dataclass(frozen=True)
class FuzzyMatch:
    start_line: int  # 1-based, inclusive
    end_line: int
    ratio: float
    exact_after_whitespace: bool


def find_fuzzy_match(
    content: str, search: str, threshold: float = FUZZY_MATCH_THRESHOLD
) -> FuzzyMatch | list[FuzzyMatch]:
    """
    Locate `search` in `content` when an exact match failed. Two passes over
    every window of len(search-lines) lines:

    1. whitespace-normalised (trailing whitespace stripped, indentation
       collapsed) equality — the common case of a model that re-indented or
       lost a trailing space;
    2. difflib similarity ≥ `threshold` — a model that paraphrased a comment
       or dropped a character.

    Returns the single `FuzzyMatch` if exactly one window qualifies, else the
    list of qualifying windows (empty = nothing close; 2+ = ambiguous). The
    caller refuses anything but a unique match: a fuzzy edit landing in the
    wrong of two similar places is worse than a clean failure.
    """
    lines = content.splitlines()
    needle_lines = search.splitlines()
    if not needle_lines or len(needle_lines) > len(lines):
        return []
    n = len(needle_lines)

    def norm(block: list[str]) -> str:
        return "\n".join(" ".join(ln.split()) for ln in block)

    needle_norm = norm(needle_lines)
    needle_text = "\n".join(needle_lines)
    exact_ws: list[FuzzyMatch] = []
    similar: list[FuzzyMatch] = []
    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seq2(needle_text)
    for i in range(len(lines) - n + 1):
        window = lines[i : i + n]
        if norm(window) == needle_norm:
            exact_ws.append(FuzzyMatch(i + 1, i + n, 1.0, True))
            continue
        matcher.set_seq1("\n".join(window))
        if matcher.quick_ratio() < threshold:
            continue
        ratio = matcher.ratio()
        if ratio >= threshold:
            similar.append(FuzzyMatch(i + 1, i + n, ratio, False))
    candidates = exact_ws or similar
    if len(candidates) == 1:
        return candidates[0]
    return candidates


async def _apply_fuzzy_search_replace(ws: Workspace, path: str, content: str, search: str, replace: str) -> ToolResult:
    found = find_fuzzy_match(content, search)
    if isinstance(found, list):
        if not found:
            return ToolResult(
                ok=False,
                error=f"The `search` text was not found in {path!r} (not even approximately). Re-read the "
                "file — it may have changed since you last saw it — and copy the exact text (including "
                "whitespace/indentation) to match.",
            )
        places = ", ".join(f"lines {m.start_line}-{m.end_line}" for m in found)
        return ToolResult(
            ok=False,
            error=f"The `search` text is not an exact match anywhere in {path!r}, and {len(found)} places "
            f"are approximately similar ({places}) — ambiguous. Re-read the file and use its exact text "
            "with enough surrounding context to match one place.",
        )

    lines = content.splitlines(keepends=True)
    trailing_newline = content.endswith("\n")
    before = "".join(lines[: found.start_line - 1])
    after = "".join(lines[found.end_line :])
    replacement = replace if (replace.endswith("\n") or (not after and not trailing_newline)) else replace + "\n"
    if not replace:
        replacement = ""
    new_content = before + replacement + after
    await ws.write_file(path, new_content)
    how = (
        "differed only in whitespace/indentation"
        if found.exact_after_whitespace
        else f"was {found.ratio:.0%} similar to the file's text"
    )
    return ToolResult(
        ok=True,
        output=f"Updated {path} ({len(new_content)} bytes) via a fuzzy match at lines "
        f"{found.start_line}-{found.end_line}: your `search` {how}. Re-read that region if you edit it again.",
    )


async def _apply_unified_diff(ws: Workspace, diff_text: str) -> ToolResult:
    patch_file = "_keystone_patch.diff"
    await ws.write_file(patch_file, diff_text if diff_text.endswith("\n") else diff_text + "\n")
    result = await ws.run(f"git apply --3way {shlex.quote(patch_file)}", check=False)
    await ws.run(f"rm -f -- {shlex.quote(patch_file)}", check=False)
    if result["exit_code"] != 0:
        return ToolResult(
            ok=False,
            error="git apply failed:\n"
            + _truncate(result.get("stderr", "") or result.get("stdout", ""), "git apply output"),
        )
    return ToolResult(ok=True, output="Patch applied.\n" + _truncate(result.get("stdout", "")))


TOOL_IMPLS: dict[str, Callable[..., Awaitable[ToolResult]]] = {
    "read_file": read_file,
    "list_dir": list_dir,
    "grep": grep,
    "get_repo_map": get_repo_map,
    "apply_patch": apply_patch,
    "run_command": run_command,
    "run_tests": run_tests,
}


async def dispatch_tool_call(
    ws: Workspace, name: str, arguments: dict[str, Any], *, touched_paths: list[str] | None = None
) -> ToolResult:
    """
    Look up and invoke a tool by name with the model-supplied `arguments`,
    for the agentic loop (`nodes/coding.py`) to call without knowing each
    tool's exact signature. A model hallucinating a tool name or passing
    arguments that don't match the schema (extra/misspelled keys, wrong
    types) is a normal, expected failure mode here — turned into a
    `ToolResult`, same as any other tool-level failure, not an exception
    that would end the whole loop over one bad call. `touched_paths` is the
    loop's own bookkeeping (files changed so far this task), which
    `run_tests(scope="related")` needs and the model never supplies.
    """
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return ToolResult(ok=False, error=f"Unknown tool {name!r}. Available tools: {sorted(TOOL_IMPLS)}")
    try:
        if name == "run_tests":
            return await run_tests(ws, touched_paths=touched_paths or [], **arguments)
        return await impl(ws, **arguments)
    except TypeError as exc:
        return ToolResult(ok=False, error=f"Invalid arguments for {name!r}: {exc}")


def touched_path(name: str, arguments: dict[str, Any], result: ToolResult) -> str | None:
    """The file path a successful apply_patch call changed, or None — for the loop's `files_touched` bookkeeping."""
    if name != "apply_patch" or not result.ok:
        return None
    return arguments.get("path")
