# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — Chat completions endpoint (OpenAI-compatible).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_scope
from src.api.middleware.rate_limiter import (
    check_request_rate,
    check_token_budget,
    record_token_usage,
)
from src.api.models.requests import CompletionRequest
from src.api.models.responses import ModelInfo, ModelListResponse
from src.config import get_settings
from src.db.models import APIKey, Tenant
from src.inference.model_router import get_model_router

router = APIRouter(prefix="/v1", tags=["inference"])

# Display order + human-readable names — not derived from model_id_map's
# dict order (Python dicts are ordered, but "declare the UX order
# explicitly" is more robust than "rely on dict insertion order forever").
_MODEL_DISPLAY_ORDER = ["coding", "coding_fallback", "reasoning"]


@router.get("/models", response_model=ModelListResponse)
async def list_models(auth: tuple = Depends(require_scope("inference"))):
    """
    OpenAI-compatible model list. `id` is exactly what a client passes as
    `"model"` in /v1/chat/completions — this is how a caller (or OpenCode's
    model picker, via cli/opencode.config.json's models block) discovers
    which roles are actually available, rather than hardcoding them.
    """
    settings = get_settings()
    model_ids = settings.model_id_map
    now = int(time.time())
    return ModelListResponse(
        data=[
            ModelInfo(id=role, created=now, root=model_ids[role]) for role in _MODEL_DISPLAY_ORDER if role in model_ids
        ]
    )


@router.post("/chat/completions")
async def chat_completions(
    req: CompletionRequest,
    request: Request,
    auth: tuple = Depends(require_scope("inference")),
):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]

    # Rate limit
    rpm = api_key.rate_limit_override or 60
    await check_request_rate(tenant.id, api_key.id, rpm)

    # Token budget pre-check
    daily_limit = api_key.daily_token_limit_override or tenant.daily_token_limit
    await check_token_budget(tenant.id, daily_limit, tenant.monthly_token_limit)

    # Route to correct model
    router_instance = get_model_router()
    client, resolved_role = await router_instance.get_client(req.model)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream_completion(client, messages, req, tenant, api_key, resolved_role),
            media_type="text/event-stream",
            headers={"X-VS-Model": resolved_role, "Cache-Control": "no-cache"},
        )

    # Non-streaming
    response = await client.complete(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
    )

    usage = response.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    if total_tokens > 0:
        await record_token_usage(tenant.id, total_tokens)

    return response


async def _stream_completion(client, messages, req, tenant, api_key, role):
    total_tokens = 0
    async for chunk in client.stream(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
    ):
        yield chunk
        # Estimate tokens from streamed content
        if "content" in chunk:
            total_tokens += 1

    if total_tokens > 0:
        await record_token_usage(tenant.id, max(total_tokens, 10))
