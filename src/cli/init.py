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


# ── Coding model backend presets (`keystone init --backend`) ─────────────────

BACKEND_CHOICES: tuple[str, ...] = ("demo-cpu", "single-gpu", "frontier-proxy", "self-hosted-gpu")

BACKEND_NOTES: dict[str, str] = {
    "demo-cpu": (
        "Coding role -> the llama.cpp demo model (Qwen2.5-Coder-0.5B, CPU, no GPU). Start it with "
        "`keystone up --with-demo-model`. Real plumbing, not answer quality: use it to verify the stack "
        "and the IDE wiring, then point the role at a GPU or the frontier proxy for real coding tasks."
    ),
    "single-gpu": (
        "Coding role -> the compose `vllm-coding` service serving Qwen2.5-Coder-7B-Instruct on one 24 GB GPU "
        "(VLLM_CODING_MODEL / _GPU_COUNT / _MAX_MODEL_LEN / _TOOL_PARSER are read by docker-compose.yml). Needs "
        "the NVIDIA container toolkit; the weights (~15 GB) download on first start."
    ),
    "frontier-proxy": (
        "Coding role -> benchmarks/frontier_proxy.py on the host. After `keystone up`, start the proxy with a "
        "real ANTHROPIC_API_KEY in your shell:\n"
        "  FRONTIER_PROXY_MODEL=claude-opus-4-6 python -m benchmarks.frontier_proxy\n"
        "(on Linux, not Docker Desktop, host.docker.internal needs an extra_hosts entry — see the comment "
        "above VLLM_CODING_URL in the .env this writes.) Not air-gapped."
    ),
    "self-hosted-gpu": (
        "Coding role -> the compose/Helm defaults (GLM-5.3-Flash on 8x80 GB; Qwen2.5-Coder-32B fallback). "
        "Change the model behind a role with VLLM_CODING_MODEL / --set vllm.coding.model."
    ),
}


def backend_overrides(backend: str) -> dict[str, str]:
    """The .env lines that point the coding role at the chosen backend — exactly what a person would
    otherwise edit by hand after reading .env.example."""
    if backend == "demo-cpu":
        return {
            "VLLM_CODING_URL": "http://demo-model:8000/v1",
            "CODING_MODEL_ID": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        }
    if backend == "single-gpu":
        return {
            "VLLM_CODING_URL": "http://vllm-coding:8000/v1",
            "CODING_MODEL_ID": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "VLLM_CODING_MODEL": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "VLLM_CODING_SERVED_MODEL_NAME": "coding",
            "VLLM_CODING_GPU_COUNT": "1",
            "VLLM_CODING_MAX_MODEL_LEN": "32768",
            "VLLM_CODING_TOOL_PARSER": "hermes",
        }
    if backend == "frontier-proxy":
        return {"VLLM_CODING_URL": "http://host.docker.internal:8090/v1"}
    if backend == "self-hosted-gpu":
        return {}
    raise ValueError(f"unknown backend {backend!r}; choose one of {BACKEND_CHOICES}")
