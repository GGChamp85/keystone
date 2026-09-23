# ADR 0004 — API keys hashed with a peppered HMAC, with rotation

**Status**: accepted and implemented (2026-09-22) — `src/api/middleware/auth.py`, `POST /v1/admin/tenants/{id}/keys/{prefix}/rotate`, `keystone keys-rotate`; verified by `tests/test_auth.py` and `tests/test_key_rotation.py`

## Context

`src/api/middleware/auth.py` stores API keys as an unsalted SHA-256 of the key. Keys are high-entropy random strings, so a rainbow table is not the threat; the threat is a leaked database plus a leaked key prefix narrowing an offline search, and the absence of any rotation path (a compromised key can only be revoked, forcing every client to be reconfigured at once).

## Decision

- `hash_key` becomes HMAC-SHA256 keyed with a per-deployment pepper (`VS_SECRET_KEY`), dual-read against the legacy SHA-256 so existing keys keep working and are re-hashed on first successful use.
- `POST /v1/admin/tenants/{id}/keys/{prefix}/rotate` mints a replacement and gives the old key a 24-hour expiry, so clients can move over without an outage.
- Prerequisite, enforced by `keystone doctor` and at startup before the cutover: `VS_SECRET_KEY` must be set and stable in production — today `src/config.py` defaults it to a random value per process, which would make every peppered hash unverifiable after a restart.

## Consequences

- One migration-free change to the hash function plus one route; the dual-read window is bounded by the rehash-on-use behaviour.
- The audit log records every rotation with the acting admin.
