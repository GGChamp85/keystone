# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — the Anthropic-compatible Messages API (`POST /v1/messages`).

A client written for the Anthropic Messages API — its SDKs, agent harnesses and IDE tools — points its
base URL at Keystone and talks to the self-hosted open-weight models (or a tenant's promoted adapter)
with no code change: same roles as `/v1/chat/completions`, same API keys, rate limits, budgets,
health-aware routing and dollar ledger (src/api/routes/_inference_common.py). The translation to and
from the OpenAI wire protocol every backend here speaks is src/inference/anthropic_compat.py; text,
tools and streaming are covered, and an unsupported block (image, document) is refused with Anthropic's
own error shape, never dropped.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.middleware.auth import require_scope
from src.api.models.requests import CountTokensRequest, MessagesRequest
from src.api.routes._inference_common import (
    account_usage,
    admit_inference_request,
    log_prompt_if_enabled,
    observe_first_token,
    usage_tokens,
)
from src.db.models import APIKey, Tenant
from src.inference.anthropic_compat import (
    AnthropicStreamTranslator,
    UnsupportedContentError,
    anthropic_messages_to_openai,
    anthropic_tool_choice_to_openai,
    anthropic_tools_to_openai,
    openai_response_to_anthropic,
)
from src.inference.health import endpoint_health
from src.orchestrator.context import count_messages_tokens, count_tokens

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/v1", tags=["inference"])


def _invalid_request(message: str) -> JSONResponse:
    """Anthropic's error envelope, so an Anthropic SDK raises its normal BadRequestError with the reason."""
    return JSONResponse(
        status_code=400, content={"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    )


def _translate_request(req: MessagesRequest) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OpenAI messages + the keyword arguments for InferenceClient.complete/stream. Raises
    UnsupportedContentError for what this gateway does not translate."""
    messages = anthropic_messages_to_openai(req.system, req.messages)
    kwargs: dict[str, Any] = {"max_tokens": req.max_tokens}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.stop_sequences:
        kwargs["stop"] = req.stop_sequences
    tools = anthropic_tools_to_openai(req.tools)
    if tools:
        kwargs["tools"] = tools
        choice = anthropic_tool_choice_to_openai(req.tool_choice)
        if choice is not None:
            kwargs["tool_choice"] = choice
    return messages, kwargs


@router.post("/messages")
async def create_message(req: MessagesRequest, auth: tuple = Depends(require_scope("inference"))):
    api_key: APIKey = auth[0]
    tenant: Tenant = auth[1]

    try:
        messages, kwargs = _translate_request(req)
    except UnsupportedContentError as exc:
        return _invalid_request(str(exc))

    last_user = next((str(m["content"]) for m in reversed(messages) if m["role"] == "user" and m["content"]), "")
    admission = await admit_inference_request(
        model=req.model, max_tokens=req.max_tokens, api_key=api_key, tenant=tenant, last_user_message=last_user
    )
    client, role, served_name = admission.client, admission.role, admission.served_name
    headers = {"X-VS-Model": role, "X-VS-Route-Decision": admission.decision_header}

    if req.stream:
        return StreamingResponse(
            _stream_message(client, messages, kwargs, req.model, tenant, api_key, role, served_name),
            media_type="text/event-stream",
            headers={**headers, "Cache-Control": "no-cache"},
        )

    try:
        response = await client.complete(messages=messages, model_override=served_name, **kwargs)
    except Exception as exc:
        endpoint_health.record_failure(role, f"{type(exc).__name__}: {exc}")
        raise
    endpoint_health.record_success(role)

    prompt_tokens, completion_tokens = usage_tokens(response.get("usage"))
    await account_usage(
        tenant=tenant, api_key=api_key, role=role, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    log_prompt_if_enabled(
        role=role, messages=messages, reply=(response.get("choices") or [{}])[0].get("message", {}).get("content")
    )
    return JSONResponse(content=openai_response_to_anthropic(response, req.model), headers=headers)


async def _stream_message(client, messages, kwargs, model, tenant, api_key, role, served_name):
    """The backend's OpenAI chunks, translated live into Anthropic stream events. Real usage comes from
    the backend's `stream_options.include_usage` final chunk; a backend that sends none is counted with
    the same tokenizer the agent uses and logged as an estimate."""
    translator = AnthropicStreamTranslator(model, estimate_input_tokens=lambda: count_messages_tokens(messages))
    started = time.monotonic()
    first_token_seen = False
    try:
        async for sse in client.stream(messages=messages, model_override=served_name, include_usage=True, **kwargs):
            raw = sse[len("data: ") :].strip() if sse.startswith("data: ") else ""
            if not raw or raw == "[DONE]":
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict):
                for event in translator.feed(chunk):
                    yield event
                if not first_token_seen and translator.text_parts:
                    first_token_seen = True
                    observe_first_token(role, time.monotonic() - started)
    except Exception as exc:
        endpoint_health.record_failure(role, f"{type(exc).__name__}: {exc}")
        raise
    endpoint_health.record_success(role)

    def _estimate() -> dict[str, int]:
        est = {
            "input_tokens": count_messages_tokens(messages),
            "output_tokens": count_tokens("".join(translator.text_parts)),
        }
        logger.warning(
            "messages.stream_usage_estimated",
            tenant_id=str(tenant.id),
            role=role,
            **est,
            detail="backend sent no usage chunk despite stream_options.include_usage; counted with tiktoken",
        )
        return est

    for event in translator.finish(fallback_usage=_estimate):
        yield event
    usage = translator.usage or {"input_tokens": 0, "output_tokens": 0}
    await account_usage(
        tenant=tenant,
        api_key=api_key,
        role=role,
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
    )
    log_prompt_if_enabled(role=role, messages=messages, reply="".join(translator.text_parts))


@router.post("/messages/count_tokens")
async def count_message_tokens(req: CountTokensRequest, auth: tuple = Depends(require_scope("inference"))):
    """Anthropic's `count_tokens`: the prompt's size as the gateway's tokenizer counts it (tiktoken, the
    same count the agent budgets with) — a close estimate for the open-weight models served here, not
    the serving model's exact tokenizer."""
    try:
        messages = anthropic_messages_to_openai(req.system, req.messages)
    except UnsupportedContentError as exc:
        return _invalid_request(str(exc))
    total = count_messages_tokens(messages)
    for tool in anthropic_tools_to_openai(req.tools) or []:
        total += count_tokens(json.dumps(tool["function"]))
    return {"input_tokens": total}
