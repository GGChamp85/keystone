# ADR 0006 — Repo-facing text is vendor-neutral

**Status**: accepted (2026-09-22)

## Context

Keystone can front a frontier model through `benchmarks/frontier_proxy.py`, and the benchmark cost model compares self-hosted GPU cost to published frontier prices. The project's public text (README, CHANGELOG, ROADMAP, `docs/`, cost-report display strings) should describe that capability without reading as an endorsement of, or dependency on, any one vendor — and without a vendor's name in prose that a customer's procurement review would flag.

## Decision

- Prose in repo-facing files says **"a frontier model"** / **"the frontier proxy"** / **"Frontier vendor A — flagship"** rather than a vendor or product name.
- Internal identifiers are unaffected: model-id strings and config keys (`claude-opus` in `benchmarks/cost_model.py`'s pricing table, the `AnthropicClient` class the proxy is built on, `FRONTIER_PROXY_MODEL`'s default value) remain what the vendor's API requires.
- Commit messages follow the same rule and carry no co-author trailers.
- Enforced by `scripts/check_vendor_neutral.sh` (a pre-commit hook over README/CHANGELOG/ROADMAP/docs), which ignores model-id tokens.

## Consequences

- Comparisons stay honest and specific in numbers (prices are real and dated in `cost_model.py`) while the display layer stays neutral.
- Adding a second frontier vendor to the proxy is a code change, not a naming change.
