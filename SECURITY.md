# Security Policy

## Supported Versions

Keystone follows semver. Security fixes are backported to the latest minor release on the current major version.

| Version | Supported |
|---|---|
| latest `1.x` | ✅ |
| `< 1.0` (pre-release) | ❌ |

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security vulnerability.**

Report privately using one of:

1. **[GitHub Security Advisories](../../security/advisories/new)** (preferred) — private to maintainers until a fix is ready.
2. Email **gauravright@gmail.com** with a description, reproduction steps, and impact assessment.

We aim to acknowledge reports within 5 business days and to have a fix or mitigation plan within 30 days for confirmed vulnerabilities, sooner for critical ones. We'll credit reporters in the release notes unless anonymity is requested.

## Scope

In scope: the Keystone application code (`src/`, `web/`), Helm chart, Docker images, and air-gap bundle scripts as shipped in this repository. A vulnerability in an upstream dependency (vLLM, Qdrant, Postgres, etc.) should generally be reported upstream, but let us know too if it's exploitable through Keystone's specific configuration of it — see `docs/LICENSES_AND_COMPLIANCE.md` for the full dependency inventory.

Out of scope: findings that require an attacker to already have root-admin API access, or that only reproduce with a deliberately insecure configuration explicitly documented as dev-only (e.g. `VS_ALLOW_UNSANDBOXED_DEV`, `make certs`'s bare self-signed cert).

## Security-relevant design notes

For context before reporting: sandboxed code execution is isolated via Firecracker microVMs (production) or gVisor (fallback), with network egress enforced host-side, never inside the sandbox — see `docs/architecture/SANDBOX_ARCHITECTURE.md`. Admin API routes require a constant-time-compared root-admin token. Secrets are designed to route through OpenBao in production. If you've found a way around any of these boundaries, that's exactly what we want reported here.
