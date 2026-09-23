#!/usr/bin/env python3
# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Generate docs/reference/configuration.md from src/config.py — every setting, its environment variable,
type, default and the comment that explains it — so the reference cannot drift from the code.

    python scripts/gen_config_reference.py            # (re)write the page
    python scripts/gen_config_reference.py --check    # exit 1 if the page is stale (CI)

Descriptions come from the source itself: a `Field(description=...)`, else the trailing comment on the
field's line, else the comment block directly above it. Sections are the `# ── Title ──` banners.
Secrets (`SecretStr`) never print their default value.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "src" / "config.py"
OUTPUT = ROOT / "docs" / "reference" / "configuration.md"

_SECTION_RE = re.compile(r"^\s*#\s*─+\s*(.+?)\s*─+\s*$")
_TRAILING_RE = re.compile(r"#\s*(.*)$")


def _clean_comment(line: str) -> str:
    return line.strip().lstrip("#").strip()


def collect_fields() -> list[dict[str, Any]]:
    sys.path.insert(0, str(ROOT))
    from src.config import Settings

    source = SOURCE.read_text()
    lines = source.splitlines()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Settings")

    # section banner per line number
    section_at: dict[int, str] = {}
    current = "Platform"
    for i, line in enumerate(lines, start=1):
        m = _SECTION_RE.match(line)
        if m:
            current = m.group(1)
        section_at[i] = current

    fields: list[dict[str, Any]] = []
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        name = node.target.id
        if name == "model_config" or name not in Settings.model_fields:
            continue
        info = Settings.model_fields[name]
        annotation = ast.get_source_segment(source, node.annotation) or ""

        description = info.description or ""
        if not description:
            first_line = lines[node.lineno - 1]
            code_part = first_line.split("#", 1)
            if len(code_part) == 2 and node.end_lineno == node.lineno:
                description = _clean_comment("#" + code_part[1])
            elif node.end_lineno and node.end_lineno > node.lineno:
                # a multi-line assignment: a trailing comment may sit on any of its lines
                for ln in range(node.lineno, node.end_lineno + 1):
                    m = _TRAILING_RE.search(lines[ln - 1].split("=", 1)[-1] if ln == node.lineno else lines[ln - 1])
                    if m and not lines[ln - 1].strip().startswith("#"):
                        description = m.group(1).strip()
                        break
        if not description:
            block: list[str] = []
            ln = node.lineno - 2
            while ln >= 0 and lines[ln].strip().startswith("#") and not _SECTION_RE.match(lines[ln]):
                block.insert(0, _clean_comment(lines[ln]))
                ln -= 1
            description = " ".join(block)

        is_secret = "SecretStr" in annotation
        if info.default_factory is not None:
            default = "(generated at startup)" if is_secret else repr(info.default_factory())
        elif is_secret:
            default = "(secret; set it)" if info.default is not None else "unset"
        elif info.default is None:
            default = "unset"
        else:
            default = repr(info.default.value if hasattr(info.default, "value") else info.default)

        fields.append(
            {
                "env": name.upper(),
                "name": name,
                "type": annotation.replace("SecretStr", "secret"),
                "default": default,
                "description": description,
                "section": section_at[node.lineno],
            }
        )
    return fields


def render(fields: list[dict[str, Any]]) -> str:
    out = [
        "# Configuration reference",
        "",
        "Every setting Keystone reads, generated from `src/config.py` by `scripts/gen_config_reference.py` "
        "(CI fails when this page is stale). Set them in `.env` (which `keystone init` writes) or as "
        "environment variables; names are case-insensitive. Every limit is `0 = unlimited` by default: "
        "capacity is bounded by what you deploy, not by a policy number.",
        "",
    ]
    by_section: dict[str, list[dict[str, Any]]] = {}
    for f in fields:
        by_section.setdefault(f["section"], []).append(f)
    for section, items in by_section.items():
        out += [f"## {section}", "", "| Variable | Type | Default | What it does |", "|---|---|---|---|"]
        for f in items:
            desc = f["description"].replace("|", "\\|")
            out.append(f"| `{f['env']}` | `{f['type']}` | `{f['default']}` | {desc} |")
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    content = render(collect_fields())
    if "--check" in argv:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            print(f"{OUTPUT.relative_to(ROOT)} is stale — run: python scripts/gen_config_reference.py", file=sys.stderr)
            return 1
        print(f"{OUTPUT.relative_to(ROOT)} is up to date")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(content)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
