# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — serving promoted adapters: the LoRA registry.

Two real mechanisms make a promoted adapter reachable by name through the
gateway (`src/inference/model_router.py` already routes a tenant's
requests to its promoted adapter's `name`):

1. **At vLLM start** — `--enable-lora --lora-modules name=path ...`.
   `write_lora_modules_manifest` renders every promoted adapter for a
   base model to a JSON file the vLLM launch reads (`docker-compose.yml`
   / Helm mount it and pass `--lora-modules` from it), so a restart
   serves everything that was promoted.
2. **Without a restart** — vLLM's runtime endpoint
   `POST /v1/load_lora_adapter {"lora_name", "lora_path"}` (enabled on
   the server by `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`).
   `load_adapter_at_runtime` calls it on the role that serves the base
   model; a server without runtime updating enabled answers 4xx, which is
   reported, not hidden — the manifest still covers the next restart.

The adapter path must be visible to the vLLM process: the same volume the
trainer wrote to (`FINETUNING_OUTPUT_DIR`) mounted into the serving pods.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import structlog
from sqlalchemy import select

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AdapterStatus, ModelAdapter

logger = structlog.get_logger(__name__)

MANIFEST_NAME = "lora_modules.json"


@dataclass(frozen=True)
class LoraModule:
    name: str
    path: str
    base_model_id: str
    tenant_id: str


async def promoted_modules(base_model_id: str | None = None) -> list[LoraModule]:
    stmt = select(ModelAdapter).where(ModelAdapter.status == AdapterStatus.PROMOTED).order_by(ModelAdapter.name)
    if base_model_id:
        stmt = stmt.where(ModelAdapter.base_model_id == base_model_id)
    async with get_db_context() as db:
        rows = (await db.execute(stmt)).scalars().all()
    return [LoraModule(r.name, r.path, r.base_model_id, str(r.tenant_id)) for r in rows]


def render_lora_modules_args(modules: list[LoraModule]) -> list[str]:
    """The exact `--lora-modules name=path ...` argv vLLM takes (empty list when nothing is promoted)."""
    if not modules:
        return []
    return ["--lora-modules", *[f"{m.name}={m.path}" for m in modules]]


async def write_lora_modules_manifest(directory: str | Path | None = None) -> Path:
    """`<FINETUNING_OUTPUT_DIR>/lora_modules.json`: every promoted adapter, grouped by base model, plus the
    ready-made argv — what the serving launch reads on (re)start."""
    modules = await promoted_modules()
    out_dir = Path(directory or get_settings().finetuning_output_dir)
    path = await asyncio.to_thread(_write_manifest_file, out_dir, modules)
    logger.info("lora_registry.manifest_written", path=str(path), adapters=len(modules))
    return path


def render_manifest(modules: list[LoraModule]) -> dict[str, Any]:
    by_base: dict[str, list[dict[str, str]]] = {}
    for m in modules:
        by_base.setdefault(m.base_model_id, []).append({"name": m.name, "path": m.path, "tenant_id": m.tenant_id})
    return {
        "modules": by_base,
        "argv": {base: render_lora_modules_args([m for m in modules if m.base_model_id == base]) for base in by_base},
    }


def _write_manifest_file(out_dir: Path, modules: list[LoraModule]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    path.write_text(json.dumps(render_manifest(modules), indent=2) + "\n")
    return path


def role_for_base_model(base_model_id: str) -> str | None:
    """Which serving role has `base_model_id` as its model, if any."""
    for role, model_id in get_settings().model_id_map.items():
        if model_id == base_model_id:
            return role
    return None


async def load_adapter_at_runtime(base_url: str, name: str, path: str, *, api_key: str | None = None) -> dict[str, Any]:
    """vLLM's `POST /v1/load_lora_adapter`. Returns {"loaded": bool, "detail": str}; never raises."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    url = base_url.rstrip("/") + "/load_lora_adapter"
    try:
        async with httpx.AsyncClient(timeout=120.0, headers=headers) as client:
            resp = await client.post(url, json={"lora_name": name, "lora_path": path})
    except httpx.HTTPError as exc:
        return {"loaded": False, "detail": f"{url}: {exc}"}
    if resp.status_code == 200:
        return {"loaded": True, "detail": f"{name} loaded on {base_url}"}
    return {
        "loaded": False,
        "detail": f"{url} -> {resp.status_code}: {resp.text[:500]} (set VLLM_ALLOW_RUNTIME_LORA_UPDATING=True on the "
        "server for live loading; the lora_modules manifest covers the next restart)",
    }


async def serve_promoted_adapter(name: str, path: str, base_model_id: str) -> dict[str, Any]:
    """Everything promote does to make an adapter servable: rewrite the manifest, then try the live load on
    the role serving its base model. The result is returned to the caller and logged, whichever way it went."""
    manifest: Path | None
    manifest_error: str | None = None
    try:
        manifest = await write_lora_modules_manifest()
    except OSError as exc:
        # The promotion is already committed; a manifest that cannot be written (FINETUNING_OUTPUT_DIR not
        # mounted on this host) is reported in the response and the log, never hidden behind a 500.
        manifest = None
        manifest_error = f"lora_modules manifest not written: {exc}"
        logger.error("lora_registry.manifest_failed", error=str(exc), adapter=name)
    role = role_for_base_model(base_model_id)
    if role is None:
        result = {"loaded": False, "detail": f"no serving role has base model {base_model_id!r} configured"}
    else:
        settings = get_settings()
        base_url = {
            "coding": settings.vllm_coding_url,
            "coding_fallback": settings.vllm_coding_fallback_url,
            "reasoning": settings.vllm_reasoning_url,
        }[role]
        result = await load_adapter_at_runtime(base_url, name, path, api_key=settings.model_api_key_map.get(role))
    logger.info("lora_registry.promoted", adapter=name, base_model=base_model_id, **result)
    return {
        **result,
        "manifest": str(manifest) if manifest else None,
        "manifest_error": manifest_error,
        "role": role,
    }
