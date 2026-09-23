#!/usr/bin/env python3
# Copyright 2024-2026 Gaurav Gupta / Keystone — Apache 2.0

"""
The task event types exist twice in TypeScript — web/src/api.ts (the SPA) and
vscode/extension/src/api.ts (the VS Code extension) — and both mirror
src/orchestrator/events.py. This check extracts the `NodeEvent`, `StepEventType`,
`StepEvent` and `TaskEvent` declarations from both files and fails when they differ,
so a field added to one client cannot silently be missing from the other.

Comparison is on the declaration text with whitespace normalised (indentation and
blank lines do not count; comments inside a declaration do, since they document
the field's meaning). Usage: `python scripts/check_ts_types_in_sync.py` — exit 1
with a unified diff per differing declaration.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB_API = ROOT / "web" / "src" / "api.ts"
EXTENSION_API = ROOT / "vscode" / "extension" / "src" / "api.ts"
DECLARATIONS = ("NodeEvent", "StepEventType", "StepEvent", "TaskEvent")


class DeclarationNotFound(ValueError):
    pass


def extract_declaration(source: str, name: str) -> str:
    """The full text of `export interface NAME {...}` or `export type NAME = ...` in `source`.

    An interface ends at its matching closing brace (nesting counted); a type alias ends at the
    first blank line or the next `export` — union members may span several lines."""
    match = re.search(rf"^export (interface|type) {re.escape(name)}\b", source, flags=re.MULTILINE)
    if match is None:
        raise DeclarationNotFound(f"no `export interface|type {name}` declaration")
    start = match.start()
    if match.group(1) == "interface":
        depth = 0
        for i in range(match.end(), len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[start : i + 1]
        raise DeclarationNotFound(f"unterminated interface {name}")
    end = re.search(r"\n\s*\n|\nexport ", source[match.end() :])
    stop = match.end() + end.start() if end else len(source)
    return source[start:stop].rstrip()


def normalise(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def compare(web_path: Path = WEB_API, extension_path: Path = EXTENSION_API) -> list[str]:
    """Human-readable problems; empty when every declaration matches."""
    problems: list[str] = []
    try:
        web = web_path.read_text()
        ext = extension_path.read_text()
    except OSError as exc:
        return [str(exc)]
    for name in DECLARATIONS:
        sides: dict[str, list[str]] = {}
        for label, source in (("web", web), ("extension", ext)):
            try:
                sides[label] = normalise(extract_declaration(source, name))
            except DeclarationNotFound as exc:
                problems.append(f"{name}: {label} ({web_path if label == 'web' else extension_path}): {exc}")
        if len(sides) < 2:
            continue
        if sides["web"] != sides["extension"]:
            diff = "\n".join(
                difflib.unified_diff(
                    sides["web"], sides["extension"], fromfile=str(web_path), tofile=str(extension_path), lineterm=""
                )
            )
            problems.append(f"{name} differs between the web UI and the VS Code extension:\n{diff}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--web", type=Path, default=WEB_API)
    parser.add_argument("--extension", type=Path, default=EXTENSION_API)
    args = parser.parse_args()
    problems = compare(args.web, args.extension)
    if problems:
        print("\n\n".join(problems), file=sys.stderr)
        print(
            f"\n{len(problems)} problem(s): keep {', '.join(DECLARATIONS)} identical in web/src/api.ts and "
            "vscode/extension/src/api.ts (both mirror src/orchestrator/events.py).",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {', '.join(DECLARATIONS)} are identical in web/src/api.ts and vscode/extension/src/api.ts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
