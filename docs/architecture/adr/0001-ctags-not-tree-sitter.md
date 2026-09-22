# ADR 0001 — Repository map: universal-ctags, not tree-sitter

**Status**: accepted (2026-09-22)

## Context

The coding agent needs a ranked outline of a repository's symbols (file, line, signature, how often each symbol is referenced) so the planner and the tool loop go straight to the right code instead of exploring from `find -maxdepth 2`. Two ways to get symbol-level structure for many languages:

- **tree-sitter** — precise ASTs per language, but each language is a compiled grammar shipped as a Python wheel (`tree-sitter-python`, `tree-sitter-go`, ...). Keystone's air-gap bundle (`airgap/`) would have to carry one wheel per supported language per platform, and the sandbox runtime image would need them installed; every new language is a packaging change.
- **universal-ctags** — one Debian package, ~50 languages, JSON output with `name`, `path`, `line`, `kind`, `scope` and, for languages that have one, `signature`. Less precise than a full AST (no call graph, regex-based parsers for some languages), but exactly the information a map needs.

The map is built inside the task's sandbox, so whatever produces it must live in `docker/sandbox-runtimes/python.Dockerfile`.

## Decision

`src/orchestrator/repo_map.py` runs `ctags --output-format=json --fields=+nKSs -R` in the sandbox, counts references with a single `grep -rohwF -f <names>` pass, and renders files in rank order within a token budget. `universal-ctags` is added to the sandbox image. If ctags is missing (a custom image), the map degrades to a labelled file list — never a silent absence.

Python-specific enrichment via the `ast` module was considered and not needed: ctags' `signature` field already carries Python (and Go, Rust, Java, C) parameter lists.

## Consequences

- One apt package to add to any custom sandbox runtime; nothing new in the wheelhouse.
- Reference counts are textual (identifier occurrences), not semantic — a common short name will over-count. Acceptable for ranking; the map is an outline, not an analysis.
- tree-sitter remains the documented fallback if regex-boundary quality on a non-Python language proves inadequate on the benchmark suite (`docs/benchmarks/`), at which point it would ship for that language only.
