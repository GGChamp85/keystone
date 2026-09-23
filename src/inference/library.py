# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the Model Library: one answer to "what models can I use here, and are they up?"

Three kinds of entry, one shape (`GET /v1/keystone/models`):

- **role** — a serving role (`coding`, `coding_fallback`, `reasoning`): the id a client passes as
  `model`, the open-weight model behind it (`settings.model_id_map`), its endpoint, and its **live
  state** from the same cached health registry the router uses (`src/inference/health.py`) — so what
  the UI shows is what the next request will hit, breaker and all, not a second opinion.
- **adapter** — this tenant's fine-tuned LoRA adapters (`ModelAdapter`): promoted ones are what the
  router serves this tenant when the base role is up, so their state follows the role's.
- **catalog** — the SLM catalog (`src/inference/catalog.py`): deployable and fine-tunable models that
  are not behind a role here, so `NOT_DEPLOYED` — with the VRAM they need, so a deploy decision has
  its number.

States: READY (probed healthy), UNHEALTHY (probe failed or breaker open), NOT_DEPLOYED (no role
serves it), TRAINING (a fine-tune job on it is running). Everything else is data from settings, the
catalog and the database — nothing is guessed.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from sqlalchemy import select

from src.config import get_settings
from src.db.connection import get_db_context
from src.db.models import AdapterStatus, FineTuneJob, ModelAdapter
from src.inference.catalog import CATALOG, CatalogEntry, find, is_known
from src.inference.client import get_inference_client
from src.inference.health import endpoint_health

logger = structlog.get_logger(__name__)

ROLE_ORDER = ("coding", "coding_fallback", "reasoning")
ROLE_PURPOSE = {
    "coding": "primary coding model: every agent task and gateway request with model='coding'",
    "coding_fallback": "used when the primary is unhealthy, and for simple tasks when complexity routing is on",
    "reasoning": "reviewer/critic: the review and planning nodes",
}


def _provider_for(base_url: str) -> str:
    host = base_url.split("//", 1)[-1].split("/", 1)[0]
    if "runpod" in host:
        return "RunPod Serverless (hosted endpoint)"
    if host.startswith(("localhost", "127.0.0.1")):
        return "local process"
    return "self-hosted vLLM"


def _entry_for(hf_id: str) -> CatalogEntry | None:
    return find(hf_id) if is_known(hf_id) else None


def _api_features(catalog_entry: CatalogEntry | None, *, lora: bool) -> dict[str, bool | None]:
    """What a client can do against this model through the gateway. `tools` is None when the model is
    not in the catalog — unknown, not assumed: native tool calling needs a vLLM parser for that model's
    format, and the agent's text tool protocol covers the rest."""
    return {
        "chat_completions": True,
        "anthropic_messages": True,
        "streaming": True,
        "tools": bool(catalog_entry.tool_parser) if catalog_entry else None,
        "json_schema": True,
        "lora_adapters": lora,
    }


async def _probe_role(role: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """State from the router's own cached health registry, plus the endpoint's real `/models` listing
    (served ids, and `max_model_len` where the server reports it) when it is up."""
    client = get_inference_client(role)
    healthy = await endpoint_health.is_healthy(role, client)
    served: list[dict[str, Any]] = []
    if healthy:
        try:
            served = await client.list_models()
        except Exception as exc:  # the health probe passed a moment ago; report, do not fail the library
            logger.warning("model_library.list_models_failed", role=role, error=str(exc))
    snap = endpoint_health.snapshot().get(role, {})
    return ("READY" if healthy else "UNHEALTHY"), snap, served


def role_adapters(adapter_rows: list[dict[str, Any]], hf_id: str) -> list[dict[str, Any]]:
    return [a for a in adapter_rows if a["base_model_id"] == hf_id]


async def build_model_library(tenant_id: Any) -> dict[str, Any]:
    settings = get_settings()
    model_ids = settings.model_id_map
    urls = settings.model_endpoint_map
    async with get_db_context() as db:
        adapters = (await db.execute(select(ModelAdapter).where(ModelAdapter.tenant_id == tenant_id))).scalars().all()
        training = (
            (
                await db.execute(
                    select(FineTuneJob.base_model).where(
                        FineTuneJob.tenant_id == tenant_id, FineTuneJob.status.in_(["pending", "running"])
                    )
                )
            )
            .scalars()
            .all()
        )
        adapter_rows: list[dict[str, Any]] = [
            {
                "name": a.name,
                "base_model_id": a.base_model_id,
                "path": a.path,
                "rank": a.rank,
                "job_type": a.job_type,
                "job_id": str(a.job_id) if a.job_id else None,
                "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                "is_default": bool(a.is_default),
                "metrics": a.metrics or {},
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in adapters
        ]
    training_bases = set(training)

    models: list[dict[str, Any]] = []
    role_state: dict[str, str] = {}
    served_bases: dict[str, str] = {}
    for role in ROLE_ORDER:
        if role not in model_ids:
            continue
        hf_id = model_ids[role]
        served_bases.setdefault(hf_id, role)
        base_url = urls[role]
        entry = _entry_for(hf_id)
        state, health, served = await _probe_role(role)
        role_state[role] = state
        served_ids = [m.get("id") for m in served if m.get("id")]
        max_len = next((m.get("max_model_len") for m in served if m.get("max_model_len")), None)
        lora_served = any(m.get("parent") for m in served)  # vLLM lists a LoRA module with `parent` = its base
        models.append(
            {
                "kind": "role",
                "id": role,
                "name": hf_id.split("/", 1)[-1],
                "purpose": ROLE_PURPOSE.get(role, ""),
                "hf_source": hf_id,
                "provider": _provider_for(base_url),
                "endpoint": base_url,
                "served_path": role,
                "served_model_ids": served_ids,
                "state": "TRAINING" if hf_id in training_bases and state != "READY" else state,
                "health": health,
                "params_b": entry.params_b if entry else None,
                "context": max_len or (entry.context if entry else None),
                "context_source": "endpoint" if max_len else ("catalog" if entry else None),
                "license": entry.license if entry else None,
                "moe": entry.moe if entry else None,
                "tool_parser": entry.tool_parser if entry else None,
                "deployment": {"vram_serve_gb": entry.vram_serve_gb if entry else None},
                "api_features": _api_features(entry, lora=lora_served or bool(role_adapters(adapter_rows, hf_id))),
                "adapters": role_adapters(adapter_rows, hf_id),
            }
        )

    for a in adapter_rows:
        base_id: str = a["base_model_id"]
        adapter_role: str | None = served_bases.get(base_id)
        promoted = a["status"] == AdapterStatus.PROMOTED.value
        state = role_state.get(adapter_role, "NOT_DEPLOYED") if (promoted and adapter_role) else "NOT_DEPLOYED"
        models.append(
            {
                "kind": "adapter",
                "id": a["name"],
                "name": a["name"],
                "purpose": f"this tenant's {a['job_type']} adapter on {base_id.split('/', 1)[-1]}"
                + (" — routed by default" if a["is_default"] and a["status"] == "promoted" else ""),
                "hf_source": a["base_model_id"],
                "provider": _provider_for(urls[adapter_role]) if adapter_role else "not served by any role",
                "endpoint": urls[adapter_role] if adapter_role else None,
                "served_path": adapter_role if adapter_role and a["is_default"] else a["name"],
                "state": state,
                "adapter_status": a["status"],
                "is_default": a["is_default"],
                "path": a["path"],
                "rank": a["rank"],
                "job_id": a["job_id"],
                "metrics": a["metrics"],
                "created_at": a["created_at"],
                "api_features": _api_features(_entry_for(base_id), lora=True) if adapter_role else {},
            }
        )

    for entry in CATALOG:
        if entry.hf_id in served_bases:
            continue
        models.append(
            {
                "kind": "catalog",
                "id": entry.hf_id,
                "name": entry.short_name,
                "purpose": entry.notes or "deployable open-weight coder; fine-tunable on your repositories",
                "hf_source": entry.hf_id,
                "provider": "catalog (not deployed here)",
                "endpoint": None,
                "served_path": None,
                "state": "TRAINING" if entry.hf_id in training_bases else "NOT_DEPLOYED",
                "params_b": entry.params_b,
                "context": entry.context,
                "license": entry.license,
                "moe": entry.moe,
                "tool_parser": entry.tool_parser,
                "deployment": {
                    "vram_serve_gb": entry.vram_serve_gb,
                    "vram_qlora_gb": entry.vram_qlora_gb,
                    "vram_lora_bf16_gb": entry.vram_lora_bf16_gb,
                },
                "api_features": _api_features(entry, lora=True),
                "adapters": role_adapters(adapter_rows, entry.hf_id),
            }
        )

    return {"generated_at": int(time.time()), "models": models}
