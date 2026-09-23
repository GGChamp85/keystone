# Contributing to Keystone

Thanks for considering a contribution. Keystone is a self-hosted, air-gapped LLM inference gateway and autonomous coding agent platform — contributions that keep it fully open-source and self-hostable (no managed SaaS dependencies) are especially welcome.

## Ground rules

- **No managed SaaS dependencies.** Every runtime dependency must be self-hostable and work with zero internet egress in production. See `docs/airgap/OFFLINE_INSTALL_RUNBOOK.md` and `docs/TELEMETRY_AUDIT.md` for what that means in practice.
- **License compliance.** Any new dependency's license must be checked against `docs/LICENSES_AND_COMPLIANCE.md` before it's added — permissive (MIT/BSD/Apache-2.0/MPL-2.0) or a documented, accepted exception (we already carry two AGPL-3.0 components with a written rationale; don't add a third without the same scrutiny).
- **No stubs, no mocks, no skeleton code.** Tests hit real infrastructure (Postgres, Redis, Qdrant, a real Docker sandbox) rather than mocking it — see `tests/` for the existing pattern.

## Dev setup

```bash
git clone https://github.com/GGChamp85/keystone.git
cd keystone
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # edit secrets
make certs              # or `make certs-ca` for a real internal CA
make build && make up
make db-migrate
```

Frontend (`web/`, only needed if you're touching the plan/execution-trace UI):

```bash
cd web && npm install && npm run dev
```

## Before opening a PR

```bash
uv sync --frozen --extra dev    # reproducible dev environment from uv.lock (or: pip install -e '.[dev]')
pre-commit install              # runs the checks below on every commit (.pre-commit-config.yaml)
ruff check src/ tests/          # lint (no --fix in CI — fix locally, then commit)
ruff format --check src/ tests/
mypy                            # type check src/ — CI requires zero errors (config: pyproject.toml [tool.mypy])
python -m pytest tests/ -v --tb=short
helm lint helm/keystone
cd web && npx tsc --noEmit && npm run build
```

All of the above run in CI (`.github/workflows/ci.yml`) — a green PR check means these all passed.

## PR checklist

- [ ] Tests added/updated for the change, and they exercise real infrastructure where the existing pattern does (not new mocks)
- [ ] `docs/` updated if behavior, config, or a deployment step changed
- [ ] No new telemetry/phone-home introduced without updating `docs/TELEMETRY_AUDIT.md`
- [ ] No new dependency without a license check (`docs/LICENSES_AND_COMPLIANCE.md`)
- [ ] Commit messages explain *why*, not just *what*

## Reporting bugs / requesting features

Use the issue templates (`.github/ISSUE_TEMPLATE/`). For security issues, do **not** open a public issue — see `SECURITY.md`.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
