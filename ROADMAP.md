# Roadmap

Public tracking of where Keystone is headed. Tracks the phased plan for bringing Keystone Agents to Codex/Cursor-class quality with company-specific fine-tuning that measurably beats plain RAG. Each phase ships behind config flags with its own tests; phases are independently shippable.

- [ ] **Phase 0 — GLM-4.6 model support**: add GLM-4.6 (MIT-licensed) as the default coding model, with Qwen2.5-Coder-32B as fallback; requires a vLLM version bump for MoE + tool-calling support.
- [ ] **Phase 1 — Real git workflow**: the background agent clones the real repo into its sandbox, runs the repo's real test suite, commits to a branch, and opens a pull request — not a JSON blob of file contents.
- [ ] **Phase 2 — Agentic loop engineering**: tool use (read/grep/run/patch) inside the coding loop instead of single-shot full-file rewrites; a review gate that fails closed; re-planning after repeated failures; context budgeting for large diffs.
- [ ] **Phase 3 — Quality gates + memory**: enforced lint/typecheck/security scanning before review; a per-repo and per-tenant memory system the agent learns from and a human can inspect, pin, and forget — exposed as terminal commands and an OpenCode plugin.
- [ ] **Phase 4 — Multi-developer + large-repo scale**: per-developer identity and concurrent task isolation; incremental, AST-aware repo indexing for monorepos.
- [ ] **Phase 5 — Fine-tuning that beats RAG**: an automated pipeline from git history and accepted agent work to training data, held-out evaluation, and per-tenant adapter serving — usable without an ML engineer.
- [ ] **Phase 6 — Benchmarks + eval, including frontier models**: a real multi-file-repo benchmark suite, run through the full agent loop, comparing Keystone (base, fine-tuned) against frontier models for an honest quality/cost testimony.
- [ ] **Phase 7 — Enterprise ease of use**: a guided `keystone init`/`up`/`doctor` CLI, a single validated config file, and a rewritten task-oriented README/docs set.
- [ ] **Phase 8 — Public repository hygiene**: this file, CI, community health files, CodeQL/Dependabot, and a documentation site.
- [ ] **Phase 9 — Cost transparency + improvement suggestions**: live per-task token/cost breakdown, pre-task cost estimates, hard budget enforcement, and an evidence-backed suggestions engine (e.g. "this repeated lint fix should become a memory").

See `CHANGELOG.md` for what has already shipped. Contributions toward any unchecked item are welcome — see `CONTRIBUTING.md`.
