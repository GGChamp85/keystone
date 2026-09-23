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
from src.api.models.requests import CompletionRequest
from src.api.models.responses import ModelInfo, ModelListResponse
from src.api.routes._inference_common import account_usage, admit_inference_request, usage_tokens
from src.config import get_settings
from src.db.models import APIKey, Tenant
from src.inference.health import endpoint_health
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

    # Ceiling, rate limit, token budget, health-aware routing, this tenant's promoted adapter —
    # shared with /v1/messages (src/api/routes/_inference_common.py).
    client, resolved_role, model_name = await admit_inference_request(
        model=req.model, max_tokens=req.max_tokens, api_key=api_key, tenant=tenant
    )

    messages = [m.to_wire() for m in req.messages]
    # Tool calling / structured output pass through to the backend unchanged — an OpenAI client's
    # `tools`, `tool_choice` and `response_format` reach the model exactly as sent (an agent harness or
    # IDE in agent mode needs the full round trip: tool_calls out, `tool` messages back in).
    if req.tools and req.response_format:
        raise HTTPException(status_code=422, detail="tools and response_format cannot be combined in one request")
    extra: dict[str, Any] = {}
    if req.tools:
        extra["tools"] = req.tools
        if req.tool_choice is not None:
            extra["tool_choice"] = req.tool_choice
    if req.response_format:
        extra["response_format"] = req.response_format

    if req.stream:
        return StreamingResponse(
            _stream_completion(client, messages, req, tenant, api_key, resolved_role, model_name, extra),
            media_type="text/event-stream",
            headers={"X-VS-Model": resolved_role, "Cache-Control": "no-cache"},
        )

    # Non-streaming — a request that errors counts toward the role's breaker; a success closes it
    try:
        response = await client.complete(
            messages=messages,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            top_p=req.top_p,
            stop=req.stop,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            model_override=model_name,
            **extra,
        )
    except Exception as exc:
        endpoint_health.record_failure(resolved_role, f"{type(exc).__name__}: {exc}")
        raise
    endpoint_health.record_success(resolved_role)

    prompt_tokens, completion_tokens = usage_tokens(response.get("usage"))
    await account_usage(
        tenant=tenant,
        api_key=api_key,
        role=resolved_role,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    return response


async def _stream_completion(client, messages, req, tenant, api_key, role, model_override=None, extra=None):
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
    stream_kwargs = {k: v for k, v in (extra or {}).items() if k != "response_format"}
    try:
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
            **stream_kwargs,
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
    except Exception as exc:
        endpoint_health.record_failure(role, f"{type(exc).__name__}: {exc}")
        raise
    endpoint_health.record_success(role)

    if usage:
        prompt_tokens, completion_tokens = usage_tokens(usage)
    else:
        prompt_tokens = count_messages_tokens(messages)
        completion_tokens = count_tokens("".join(content_parts))
        logger.warning(
            "completions.stream_usage_estimated",
            tenant_id=str(tenant.id),
            role=role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            detail="backend sent no usage chunk despite stream_options.include_usage; counted with tiktoken",
        )
    await account_usage(
        tenant=tenant, api_key=api_key, role=role, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
