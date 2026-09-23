# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — Chat completions endpoint (OpenAI-compatible).
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.middleware.auth import require_scope
from src.api.middleware.rate_limiter import (
    check_request_rate,
    check_token_budget,
    record_token_usage,
)
from src.api.models.requests import CompletionRequest
from src.api.models.responses import ModelInfo, ModelListResponse
from src.billing.ledger import record_usage
from src.config import get_settings
from src.db.models import APIKey, Tenant
from src.inference.model_router import get_model_router, resolve_model_name_for_client
from src.orchestrator.context import count_messages_tokens, count_tokens

logger = structlog.get_logger(__name__)

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

    # Deployment-wide ceiling (MAX_TOKENS_PER_REQUEST) — the request model's
    # own bound is the protocol maximum, not what this deployment is willing
    # to serve; rejected up front rather than silently clamped.
    settings = get_settings()
    max_tokens_ceiling = settings.max_tokens_per_request  # 0 = no deployment ceiling; the model's own limit applies
    if max_tokens_ceiling > 0 and req.max_tokens > max_tokens_ceiling:
        raise HTTPException(
            status_code=422,
            detail=f"max_tokens={req.max_tokens} exceeds this deployment's limit of {max_tokens_ceiling}",
        )

    # Rate limit
    rpm = api_key.rate_limit_override or settings.default_requests_per_minute  # 0 = unlimited
    await check_request_rate(tenant.id, api_key.id, rpm)

    # Token budget pre-check
    daily_limit = api_key.daily_token_limit_override or tenant.daily_token_limit
    await check_token_budget(tenant.id, daily_limit, tenant.monthly_token_limit)

    # Route to correct model
    router_instance = get_model_router()
    client, resolved_role = await router_instance.get_client(req.model)
    # This tenant's own promoted LoRA adapter for this base model, if any —
    # the direct customer-facing gateway, so this is the highest-value
    # place adapter routing applies (src/inference/model_router.py).
    model_name = await resolve_model_name_for_client(client, tenant.id)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        return StreamingResponse(
            _stream_completion(client, messages, req, tenant, api_key, resolved_role, model_name),
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
        model_override=model_name,
    )

    usage = response.get("usage", {})
    total_tokens = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
    if total_tokens > 0:
        await record_token_usage(tenant.id, total_tokens, model_role=resolved_role)
        await record_usage(
            tenant.id,
            model_role=resolved_role,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            api_key_id=api_key.id,
        )

    return response


async def _stream_completion(client, messages, req, tenant, api_key, role, model_override=None):
    """
    Streams the backend's SSE through unchanged and bills REAL usage: the
    backend is always asked for `stream_options.include_usage` (vLLM and
    llama.cpp both honor it), the usage-only final chunk is recorded — and
    forwarded only if the caller asked for it, since an OpenAI client that
    did not may not expect a chunk with no choices. If the backend sends no
    usage at all, prompt and completion tokens are counted with the same
    tokenizer the agent uses and logged as an estimate, never as ~1 token
    per chunk (the previous accounting under-billed by orders of magnitude).
    """
    forward_usage = bool((req.stream_options or {}).get("include_usage"))
    usage: dict[str, Any] | None = None
    content_parts: list[str] = []
    async for sse in client.stream(
        messages=messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
        stop=req.stop,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        model_override=model_override,
        include_usage=True,
    ):
        raw = sse[len("data: ") :].strip() if sse.startswith("data: ") else ""
        if raw and raw != "[DONE]":
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                chunk = None
            if isinstance(chunk, dict):
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    text = (choice.get("delta") or {}).get("content")
                    if text:
                        content_parts.append(text)
                if not chunk.get("choices") and chunk.get("usage") and not forward_usage:
                    continue  # the backend's usage-only chunk was for our accounting; the client did not ask for it
        yield sse

    if usage:
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    else:
        prompt_tokens = count_messages_tokens(messages)
        completion_tokens = count_tokens("".join(content_parts))
        total_tokens = prompt_tokens + completion_tokens
        logger.warning(
            "completions.stream_usage_estimated",
            tenant_id=str(tenant.id),
            role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            detail="backend sent no usage chunk despite stream_options.include_usage; counted with tiktoken",
        )
    if total_tokens > 0:
        await record_token_usage(tenant.id, total_tokens, model_role=role)
        await record_usage(
            tenant.id,
            model_role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            api_key_id=api_key.id,
        )
