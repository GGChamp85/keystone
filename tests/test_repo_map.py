# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
src/orchestrator/repo_map.py — the pure parse/rank/render steps against
real ctags JSON (captured from universal-ctags, not invented), plus one
real end-to-end build inside a live sandbox whose image carries ctags
(requires SANDBOX_DAEMON_URL; skipped otherwise, like test_tool_impl.py).
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from src.orchestrator.context import count_tokens
from src.orchestrator.repo_map import (
    RepoMap,
    Symbol,
    build_repo_map,
    parse_ctags_json,
    parse_reference_counts,
    rank,
    render,
)

# Real `ctags --output-format=json --fields=+nKSs` records (one JSON object per line) for a
# two-file Python project — the field names and kinds are exactly what universal-ctags emits.
_CTAGS_RECORDS = [
    {
        "_type": "tag",
        "name": "Greeter",
        "path": "./app.py",
        "pattern": "/^class Greeter:$/",
        "line": 1,
        "kind": "class",
    },
    {
        "_type": "tag",
        "name": "greet",
        "path": "./app.py",
        "pattern": "/^    def greet(self, name):$/",
        "line": 2,
        "kind": "member",
        "scope": "Greeter",
        "scopeKind": "class",
        "signature": "(self, name)",
    },
    {
        "_type": "tag",
        "name": "main",
        "path": "./app.py",
        "pattern": "/^def main():$/",
        "line": 6,
        "kind": "function",
        "signature": "()",
    },
    {"_type": "tag", "name": "x", "path": "./app.py", "pattern": "/^x = 1$/", "line": 9, "kind": "variable"},
    {
        "_type": "tag",
        "name": "helper",
        "path": "./util/helpers.py",
        "pattern": "/^def helper():$/",
        "line": 1,
        "kind": "function",
        "signature": "()",
    },
    {"_type": "ptag", "name": "JSON_OUTPUT_VERSION", "path": "1.0", "pattern": "in development"},
]
_CTAGS_OUTPUT = "\n".join(json.dumps(r) for r in _CTAGS_RECORDS) + "\nnot json at all\n"


def test_parse_keeps_structural_tags_and_drops_variables_and_pseudo_tags():
    symbols = parse_ctags_json(_CTAGS_OUTPUT)
    names = [(s.name, s.kind, s.path, s.line) for s in symbols]
    assert names == [
        ("Greeter", "class", "app.py", 1),
        ("greet", "member", "app.py", 2),
        ("main", "function", "app.py", 6),
        ("helper", "function", "util/helpers.py", 1),
    ]
    greet = symbols[1]
    assert greet.signature == "(self, name)"
    assert greet.scope == "Greeter"


def test_parse_reference_counts_reads_uniq_c_output():
    assert parse_reference_counts("   3 greet\n  12 Greeter\ngarbage\n") == {"greet": 3, "Greeter": 12}


def test_rank_orders_files_and_symbols_by_references_excluding_the_definition():
    symbols = parse_ctags_json(_CTAGS_OUTPUT)
    # helper is mentioned 6 times (1 definition + 5 uses); Greeter 3 times; greet twice; main once.
    repo_map = rank(symbols, {"helper": 6, "Greeter": 3, "greet": 2, "main": 1})
    assert repo_map.ranked_paths == ["util/helpers.py", "app.py"]
    app_syms = repo_map.files["app.py"]
    assert [s.name for s in app_syms] == ["Greeter", "greet", "main"]
    assert [s.refs for s in app_syms] == [2, 1, 0]
    assert repo_map.files["util/helpers.py"][0].refs == 5


def test_render_shows_signatures_scope_indentation_and_reference_counts():
    repo_map = rank(parse_ctags_json(_CTAGS_OUTPUT), {"helper": 6, "Greeter": 3})
    text = render(repo_map, max_tokens=2000)
    assert text.startswith("Repository map (2 files, 4 symbols;")
    assert "util/helpers.py:\n  1: function helper()  (refs 5)" in text
    assert "app.py:\n  1: class Greeter  (refs 2)\n    2: member greet(self, name)\n  6: function main()" in text
    assert "more files" not in text


def test_render_respects_the_token_budget_and_says_what_it_left_out():
    many = [Symbol(name=f"fn{i}", path=f"pkg/mod{i}.py", line=1, kind="function", signature="()") for i in range(200)]
    repo_map = rank(many, {})
    text = render(repo_map, max_tokens=300)
    assert count_tokens(text) <= 300 + 40  # the trailer line is added after the budget check
    assert "more files not shown (token budget 300)" in text
    # The first-ranked file is always present even under a tiny budget.
    assert "pkg/mod0.py" in text


def test_render_caps_symbols_per_file_and_keeps_source_order():
    syms = [Symbol(name=f"f{i}", path="big.py", line=i + 1, kind="function") for i in range(50)]
    text = render(rank(syms, {}), max_tokens=5000)
    assert "... +10 more symbols" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("  ") and ": function" in ln]
    assert [int(ln.split(":")[0]) for ln in lines] == sorted(int(ln.split(":")[0]) for ln in lines)


def test_render_without_ctags_is_a_labelled_file_list():
    repo_map = RepoMap(files={"a.py": [], "b/c.go": []}, ranked_paths=["a.py", "b/c.go"], ctags_available=False)
    text = render(repo_map)
    assert text.splitlines()[0].startswith("Repository map (ctags is not available")
    assert "a.py" in text and "b/c.go" in text


# ── real build inside a live sandbox ──────────────────────────

requires_sandbox = pytest.mark.skipif(
    not os.environ.get("SANDBOX_DAEMON_URL"),
    reason="Needs SANDBOX_DAEMON_URL pointed at a running keystoned with the sandbox image built",
)


@pytest.fixture
async def ws():
    from src.orchestrator.workspace import Workspace
    from src.sandbox.manager import SandboxManager

    manager = SandboxManager()
    workspace = Workspace(manager, task_id=f"repo-map-test-{uuid.uuid4().hex[:8]}")
    await workspace.ensure_sandbox(network_enabled=False)
    await workspace.run("mkdir -p /workspace/repo/util", cwd="/")
    await workspace.write_file(
        "app.py",
        "from util.helpers import helper\n\n\nclass Greeter:\n    def greet(self, name):\n"
        "        return helper(name)\n\n\ndef main():\n    return Greeter().greet('x')\n",
    )
    await workspace.write_file("util/helpers.py", "def helper(name):\n    return f'hello {name}'\n")
    await workspace.write_file("util/__init__.py", "")
    await workspace.run(
        "git init -q && git -c user.name=t -c user.email=t@t.dev add -A "
        "&& git -c user.name=t -c user.email=t@t.dev commit -q -m init"
    )
    try:
        yield workspace
    finally:
        await workspace.close()


@requires_sandbox
async def test_build_repo_map_runs_real_ctags_in_the_sandbox(ws):
    text = await build_repo_map(ws, max_tokens=2000)
    assert text.startswith("Repository map ("), text
    assert "ctags is not available" not in text, "the sandbox image must carry universal-ctags"
    # Real symbols, real signatures, real line numbers.
    assert "app.py:" in text
    assert "4: class Greeter" in text
    assert "5: member greet(self, name)" in text or "5: method greet(self, name)" in text
    assert "9: function main()" in text
    assert "util/helpers.py:" in text
    assert "1: function helper(name)" in text
    # helper is defined once and referenced twice (import + call) -> refs 2; it should outrank main (refs 0).
    assert "helper(name)  (refs 2)" in text
    assert "function main()  (refs" not in text
