# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — repository map.

A compact, ranked outline of a repository's symbols (classes, functions,
methods, types — with file, line and signature) that fits in a fixed token
budget, so the planner and the coding loop start with a picture of the
whole codebase instead of `find -maxdepth 2` and guesswork. This is the
single biggest structural difference between a tool loop that explores
blindly and one that goes straight to the right file.

How it's built (all inside the task's real sandbox, via `Workspace.run`):

1. `universal-ctags` emits one JSON object per symbol for every language it
   knows (Python, Go, Rust, TypeScript/JavaScript, Java, C/C++, ...), with
   the signature field where the language has one. One apt package in the
   sandbox image — no compiled grammar wheels to carry into the air-gap
   bundle, which is why this is ctags and not tree-sitter (ADR 0001).
2. Every symbol name is counted across the repository with one `grep -F`
   pass (Aho-Corasick, so thousands of names cost one file scan). A symbol
   referenced from many places is more important to show than one nobody
   calls; a file whose symbols are widely referenced ranks above a leaf.
3. Files are rendered in rank order until the token budget is spent, so
   the most-connected code is always in the map and the tail is summarised
   as "N more files".

If ctags is missing from the sandbox image (an older image, or a custom
runtime), the map degrades to a ranked file list and says so in its first
line — a weaker map, never a silent one.

The parse/rank/render steps are pure functions over ctags' output, tested
without a sandbox; only `build_repo_map` touches the workspace.
"""

from __future__ import annotations

import json
import shlex
from collections import defaultdict
from dataclasses import dataclass, field

from src.orchestrator.context import count_tokens
from src.orchestrator.workspace import Workspace

DEFAULT_MAP_TOKENS = 4_000
MAX_SYMBOLS_PER_FILE = 40

# ctags "kind" long names worth showing. Locals, variables, imports and
# similar noise are dropped — a map is an outline, not a concordance.
_STRUCTURAL_KINDS = frozenset(
    {
        "class",
        "function",
        "method",
        "member",
        "struct",
        "interface",
        "type",
        "typedef",
        "func",
        "enum",
        "trait",
        "impl",
        "module",
        "namespace",
        "constructor",
        "property",
        "alias",
    }
)

_EXCLUDES = (".git", "node_modules", ".venv", "venv", "dist", "build", "__pycache__", ".tox", ".mypy_cache", "target")

_CTAGS_CMD = (
    "ctags --output-format=json --fields=+nKSs -R "
    + " ".join(f"--exclude={shlex.quote(e)}" for e in _EXCLUDES)
    + " . 2>/dev/null"
)
_NAMES_FILE = "_keystone_symbol_names.txt"


@dataclass(frozen=True)
class Symbol:
    name: str
    path: str
    line: int
    kind: str
    signature: str = ""
    scope: str = ""
    refs: int = 0  # references across the repo, excluding the definition itself


@dataclass
class RepoMap:
    files: dict[str, list[Symbol]] = field(default_factory=dict)  # path -> symbols, ranked
    ranked_paths: list[str] = field(default_factory=list)
    ctags_available: bool = True

    @property
    def symbol_count(self) -> int:
        return sum(len(s) for s in self.files.values())


# ── pure steps ────────────────────────────────────────────────


def parse_ctags_json(output: str) -> list[Symbol]:
    """One `Symbol` per structural tag in ctags' `--output-format=json` output (one JSON object per line)."""
    symbols: list[Symbol] = []
    for raw in output.splitlines():
        raw = raw.strip()
        if not raw.startswith("{"):
            continue
        try:
            tag = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if tag.get("_type") != "tag" or tag.get("kind") not in _STRUCTURAL_KINDS:
            continue
        name = tag.get("name")
        path = tag.get("path")
        line = tag.get("line")
        if not name or not path or not isinstance(line, int):
            continue
        symbols.append(
            Symbol(
                name=str(name),
                path=str(path).removeprefix("./"),
                line=line,
                kind=str(tag["kind"]),
                signature=str(tag.get("signature") or ""),
                scope=str(tag.get("scope") or ""),
            )
        )
    return symbols


def parse_reference_counts(uniq_c_output: str) -> dict[str, int]:
    """`sort | uniq -c` output ("  12 name") -> {name: 12}."""
    counts: dict[str, int] = {}
    for raw in uniq_c_output.splitlines():
        parts = raw.split(None, 1)
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        counts[parts[1].strip()] = int(parts[0])
    return counts


def rank(symbols: list[Symbol], reference_counts: dict[str, int]) -> RepoMap:
    """Attach reference counts, then rank files by the total references to their symbols
    (ties broken by path) and symbols within a file by references, then line."""
    by_file: dict[str, list[Symbol]] = defaultdict(list)
    for sym in symbols:
        # The definition itself is one occurrence of the name; don't count it as a reference.
        refs = max(0, reference_counts.get(sym.name, 0) - 1)
        by_file[sym.path].append(Symbol(sym.name, sym.path, sym.line, sym.kind, sym.signature, sym.scope, refs=refs))

    files: dict[str, list[Symbol]] = {}
    for path, syms in by_file.items():
        files[path] = sorted(syms, key=lambda s: (-s.refs, s.line))

    def file_score(path: str) -> tuple[int, str]:
        return (-sum(s.refs for s in files[path]), path)

    ranked_paths = sorted(files, key=file_score)
    return RepoMap(files=files, ranked_paths=ranked_paths)


def _render_symbol(sym: Symbol) -> str:
    indent = "    " if sym.scope else "  "
    refs = f"  (refs {sym.refs})" if sym.refs else ""
    return f"{indent}{sym.line}: {sym.kind} {sym.name}{sym.signature}{refs}"


def render(repo_map: RepoMap, max_tokens: int = DEFAULT_MAP_TOKENS) -> str:
    """Files in rank order until `max_tokens` (tiktoken-counted) is spent; the remainder is summarised."""
    if not repo_map.ctags_available:
        header = "Repository map (ctags is not available in this sandbox image — file list only, no symbols):"
    else:
        header = (
            f"Repository map ({len(repo_map.ranked_paths)} files, {repo_map.symbol_count} symbols; "
            "(refs N) = references across the repo):"
        )
    lines = [header]
    used = count_tokens(header)
    shown = 0
    for path in repo_map.ranked_paths:
        syms = repo_map.files.get(path, [])
        block = [path] if not syms else [f"{path}:"]
        # Keep source order within the per-file cap so the outline reads top to bottom;
        # the cap itself keeps the most-referenced symbols (ranked order) when trimming.
        kept = sorted(syms[:MAX_SYMBOLS_PER_FILE], key=lambda s: s.line)
        block.extend(_render_symbol(s) for s in kept)
        if len(syms) > MAX_SYMBOLS_PER_FILE:
            block.append(f"    ... +{len(syms) - MAX_SYMBOLS_PER_FILE} more symbols")
        text = "\n".join(block)
        cost = count_tokens(text)
        if used + cost > max_tokens and shown > 0:
            break
        lines.append(text)
        used += cost
        shown += 1
    remaining = len(repo_map.ranked_paths) - shown
    if remaining > 0:
        lines.append(
            f"... {remaining} more files not shown (token budget {max_tokens}); use list_dir/grep to find them"
        )
    return "\n".join(lines)


# ── the real operation ────────────────────────────────────────


async def build_repo_map(ws: Workspace, max_tokens: int = DEFAULT_MAP_TOKENS) -> str:
    """Run ctags + one grep pass inside the sandbox and render the ranked map within `max_tokens`."""
    ctags_result = await ws.run(_CTAGS_CMD, timeout=120, check=False)
    if ctags_result["exit_code"] == 127 or "not found" in (ctags_result.get("stderr") or ""):
        return render(await _file_list_fallback(ws), max_tokens)

    symbols = parse_ctags_json(ctags_result.get("stdout", ""))
    if not symbols:
        return render(await _file_list_fallback(ws, ctags_available=True), max_tokens)

    names = sorted({s.name for s in symbols})
    await ws.write_file(_NAMES_FILE, "\n".join(names) + "\n")
    exclude = " ".join(f"--exclude-dir={shlex.quote(e)}" for e in _EXCLUDES)
    grep_cmd = f"grep -rohwF -f {_NAMES_FILE} {exclude} --exclude={_NAMES_FILE} . | sort | uniq -c"
    grep_result = await ws.run(grep_cmd, timeout=120, check=False)
    await ws.run(f"rm -f -- {_NAMES_FILE}", check=False)
    counts = parse_reference_counts(grep_result.get("stdout", "")) if grep_result["exit_code"] in (0, 1) else {}

    return render(rank(symbols, counts), max_tokens)


async def _file_list_fallback(ws: Workspace, ctags_available: bool = False) -> RepoMap:
    result = await ws.run("git ls-files 2>/dev/null || find . -type f -not -path '*/.git/*'", check=False)
    paths = [p.strip().removeprefix("./") for p in result.get("stdout", "").splitlines() if p.strip()]
    return RepoMap(files={p: [] for p in paths}, ranked_paths=sorted(paths), ctags_available=ctags_available)
