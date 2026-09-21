# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — tool schemas.

OpenAI-compatible function-calling shape (verified against vLLM's real
`/chat/completions` `tools` parameter, tested in tests/test_inference_client.py
against a real server), so the same schema works for both
`protocol.NativeToolProtocol` (passed straight through as `tools=`) and
`protocol.TextToolProtocol` (rendered into the prompt as a description of
the callable functions).

Five tools, matching the plan: three read-only exploration tools
(read_file/list_dir/grep) the model can call freely to understand a repo
before editing, one write tool (apply_patch) that's the *only* way the
model changes a file's content, and one broad tool (run_command) for
everything else (installing deps, running a linter directly, etc.) —
gated more heavily than the others since it's the least constrained.
"""

from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's full contents from the repository working tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the repository root, e.g. 'src/app.py'.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories under a path in the repository, up to a given depth.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the repository root. Default '.' (repo root).",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "How many directory levels to descend. Default 2.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search file contents in the repository for a pattern (regex), "
                "returning matching lines with file:line prefixes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for (grep -E syntax).",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path or glob to search under. Default '.' (whole repo).",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search. Default false.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Change a file's content. Preferred mode: give `search` (must match the file's current "
                "content EXACTLY, including whitespace) and `replace` (what to put there instead) — "
                "use this for most edits, it's more reliable than a diff because there's no line-number "
                "drift to get wrong. For a brand-new file, omit `search` and set `create=true`. For "
                "multi-hunk or multi-file changes in one call, give `unified_diff` instead (a real "
                "`git diff`-format patch) — it's applied with `git apply --3way`, which tolerates minor "
                "context drift; a hard conflict is returned as a structured failure, not a silent partial edit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "File path relative to the repo root. Required for search/replace mode; "
                            "omit when using unified_diff (the diff names its own paths)."
                        ),
                    },
                    "search": {
                        "type": "string",
                        "description": "Exact text to find in the file (search/replace mode).",
                    },
                    "replace": {
                        "type": "string",
                        "description": "Text to replace it with (search/replace mode).",
                    },
                    "create": {
                        "type": "boolean",
                        "description": "Create `path` as a new file with `replace` as its content. Default false.",
                    },
                    "delete": {
                        "type": "boolean",
                        "description": "Delete `path` instead of editing it. Default false.",
                    },
                    "unified_diff": {
                        "type": "string",
                        "description": "A full unified diff (git diff format) to apply instead of search/replace.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the repository working tree (e.g. to install a dependency, run "
                "a linter directly, or inspect something list_dir/grep can't). Prefer the repo's real "
                "test command (already run automatically after your changes) over reinventing it here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."},
                    "timeout_seconds": {"type": "integer", "description": "Max seconds to allow. Default 60, max 300."},
                },
                "required": ["command"],
            },
        },
    },
]

TOOL_NAMES = frozenset(t["function"]["name"] for t in TOOL_SCHEMAS)


def render_tools_as_text(schemas: list[dict] = TOOL_SCHEMAS) -> str:
    """
    Text-protocol rendering (protocol.TextToolProtocol) — for a model role
    without native tool-calling support (see src/inference/config.py's
    `tool_protocol`). Describes each tool and the exact fenced-JSON call
    format the model must use, parsed back out by
    protocol.TextToolProtocol.parse_tool_calls.
    """
    lines = [
        "You have access to the following tools. To call one, respond with "
        "ONLY a fenced code block like this and nothing else:",
        '```tool_call\n{"name": "<tool_name>", "arguments": {...}}\n```',
        "",
        "Available tools:",
    ]
    for t in schemas:
        fn = t["function"]
        lines.append(f"- {fn['name']}({', '.join(fn['parameters'].get('properties', {}).keys())}): {fn['description']}")
    return "\n".join(lines)
