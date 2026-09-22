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

Seven tools: four read-only exploration tools (read_file with line ranges,
list_dir, grep, get_repo_map — the ranked symbol outline from
src/orchestrator/repo_map.py) the model can call freely to understand a
repo before editing, one write tool (apply_patch) that's the *only* way the
model changes a file's content, one verification tool (run_tests — the
repo's real suite, related-tests-first) so the model checks its own work
before signaling done, and one broad tool (run_command) for everything
else (running a linter directly, etc.) — gated more heavily than the
others since it's the least constrained.
"""

from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the repository working tree — the whole file, or a line range "
                "(returned with line numbers). Use a range for large files instead of reading everything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the repository root, e.g. 'src/app.py'.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return (1-based, inclusive). Omit for the whole file.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to return (inclusive). Omit for 'to the end of the file'.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_map",
            "description": (
                "A ranked outline of the repository's symbols (classes, functions, methods with file, line "
                "and signature), most-referenced code first, within a token budget. The first call's "
                "result is already in your context; call again with a larger max_tokens to see more."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "max_tokens": {
                        "type": "integer",
                        "description": "Budget for the map in tokens. Default 4000, max 16000.",
                    },
                },
                "required": [],
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
                "Change a file's content. Preferred mode: give `search` (the file's current text — an "
                "exact match is best; a match that differs only in whitespace or is ≥92% similar is "
                "accepted when it is unambiguous, and the result says so) and `replace` (what to put "
                "there instead) — use this for most edits, it's more reliable than a diff because there's "
                "no line-number drift to get wrong. For a brand-new file, omit `search` and set `create=true`. For "
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
            "name": "run_tests",
            "description": (
                "Run the repository's own test suite (auto-detected: pytest, npm test, go test, cargo test). "
                "scope='related' runs only the tests covering the files you have changed so far — fast, "
                "use it after every meaningful edit; scope='all' runs everything — use it once before you "
                "finish. The exit code and the tail of the output are returned; a failure is information "
                "to act on, not an error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["related", "all"],
                        "description": "'related' (default) or 'all'.",
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
                "Run a shell command in the repository working tree (e.g. to run a linter directly, or "
                "inspect something list_dir/grep can't). Dependencies were installed automatically after "
                "the clone; use run_tests, not this, to run the test suite."
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
