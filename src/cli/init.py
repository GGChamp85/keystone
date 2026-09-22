# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — `keystone init` guided setup.

Writes a real .env file for `make up` to actually use — not a separate
keystone.yaml that then has to be translated into .env, since .env
(pydantic-settings, src/config.py) already is Keystone's single real
config surface; adding a second format on top of it would just be
something else to keep in sync; see docker-compose.yml/.env.example for
the schema this fills in.

`render_env_file` is the pure, real logic (line-based override of
.env.example, preserving every comment and unrelated line) — fully
testable with no interactivity. `prompt_for_overrides` is the interactive
wrapper around it.
"""

from __future__ import annotations

import secrets
from pathlib import Path


def generate_secret() -> str:
    return secrets.token_hex(32)


def render_env_file(template_text: str, overrides: dict[str, str]) -> str:
    """
    Real, minimal templating: every `KEY=...` line in `template_text`
    whose KEY is in `overrides` gets replaced with the real value;
    everything else (comments, blank lines, unrelated settings) passes
    through unchanged. Override keys with no matching line in the
    template are appended at the end, under their own labeled section —
    so this never silently drops a real value the caller asked for.
    """
    lines = template_text.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0]
        if key in overrides:
            out.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            out.append(line)

    missing = [k for k in overrides if k not in seen]
    if missing:
        out.append("")
        out.append("# ---------- Added by `keystone init` ----------")
        out.extend(f"{k}={overrides[k]}" for k in missing)

    return "\n".join(out) + "\n"


def find_env_template(start: Path | None = None) -> Path | None:
    """Looks for .env.example starting from `start` (default: CWD) and
    walking up — matches how `make`/`docker compose` already assume a
    repo-root CWD, rather than bundling a copy for a pip-installed CLI
    running somewhere else entirely."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        template = candidate / ".env.example"
        if template.exists():
            return template
    return None


def default_overrides(auto_secrets: bool = True) -> dict[str, str]:
    """
    The minimal real set every deployment needs regardless of any
    interactive choice — secrets, generated fresh rather than left as
    .env.example's placeholder values (see keystone doctor's
    check_secrets, which flags exactly these placeholders).

    DATABASE_URL/REDIS_URL are regenerated alongside POSTGRES_PASSWORD/
    REDIS_PASSWORD, not independently — .env.example embeds the same
    password in both the bare *_PASSWORD key (what docker-compose.yml
    actually interpolates into its own computed DATABASE_URL/REDIS_URL
    for the compose-managed app/worker containers) and a literal
    DATABASE_URL/REDIS_URL line (what a directly-run process reads via
    pydantic-settings' env file). Overriding only one half would leave
    the two inconsistent for anyone not going through compose.
    """
    if not auto_secrets:
        return {}
    postgres_password = generate_secret()
    redis_password = generate_secret()
    return {
        "POSTGRES_PASSWORD": postgres_password,
        "DATABASE_URL": f"postgresql+asyncpg://keystone:{postgres_password}@postgres:5432/keystone",
        "REDIS_PASSWORD": redis_password,
        "REDIS_URL": f"redis://:{redis_password}@redis:6379/0",
        "QDRANT_API_KEY": generate_secret(),
        "KEYSTONE_ROOT_ADMIN_TOKEN": generate_secret(),
        "VS_SECRET_KEY": generate_secret(),
    }
