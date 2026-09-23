# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — OpenAI-compatible proxy in front of a real frontier model.

src/inference/client.py's InferenceClient (the same client every real
orchestrator node — planning, coding, review, quality, tool_execution —
already uses) speaks exactly one wire protocol: POST {base_url}/chat/completions
with an OpenAI chat-completions payload, tools in OpenAI's function-calling
shape, and (for chat_structured()) vLLM's `response_format: json_schema`
guided-decoding convention. It has no idea whether the far end is vLLM or
something else.

This module IS something else: a real FastAPI server that translates that
exact wire protocol to and from the real Anthropic Messages API, using the
real `anthropic` SDK — no mocking, no stub responses. Point VLLM_CODING_URL
(or any model role's *_URL) at this server's /v1 and the *entire real agent
loop* — the actual LangGraph nodes, actual tool-calling protocol, actual
structured-output retries — runs against a real frontier model, not just
the standalone single-completion benchmarks/model_clients.py path. This is
what benchmarks/agent_runner.py uses to drive a real, full agentic task
against a frontier model for Phase 6's repo-task suite.

response_format (vLLM's guided JSON decoding) has no Anthropic equivalent,
so it's implemented via the well-known "forced single tool call" technique:
a synthetic tool whose input_schema IS the requested JSON schema, with
tool_choice forced to it — the tool call's `input` (already a parsed dict)
is what would have been the guided-decoded JSON content.

Streaming only translates text deltas (tool-call streaming is not
translated) — sufficient for src/api/routes/completions.py's streaming
gateway endpoint; no orchestrator node actually streams (confirmed: only
completions.py and the CLI call InferenceClient.stream()).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import anthropic
import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.inference.anthropic_compat import (
    STRUCTURED_TOOL_NAME as _STRUCTURED_TOOL_NAME,
)
from src.inference.anthropic_compat import (
    anthropic_response_to_openai as _anthropic_response_to_openai,
)
from src.inference.anthropic_compat import (
    openai_messages_to_anthropic as _openai_messages_to_anthropic,
)
from src.inference.anthropic_compat import (
    openai_tool_choice_to_anthropic as _openai_tool_choice_to_anthropic,
)
from src.inference.anthropic_compat import (
    openai_tools_to_anthropic as _openai_tools_to_anthropic,
)
from src.inference.anthropic_compat import (
    redact_pii_in_place as _redact_pii_in_place,
)

logger = structlog.get_logger(__name__)


# Whatever `model` string the caller's InferenceClient sends (e.g.
# "zai-org/GLM-5.3-Flash", or a promoted adapter's served name) is ignored —
# every request this proxy receives is served by this one real Anthropic
# model, configured once for the whole process.
_TARGET_MODEL = os.environ.get("FRONTIER_PROXY_MODEL", "claude-opus-4-6")

# Off by default: this is the one real network-egress boundary in the
# whole platform where request content leaves the self-hosted network for
# a third-party API, so it's the highest-value real place to redact PII
# before it crosses that boundary — but redaction is a real behavior
# change (a matched span becomes a placeholder before the model ever sees
# it), which can alter a legitimate task's outcome if the task genuinely
# needs the real value (an email/phone/IP literal in a test fixture, for
# instance). Left as an explicit opt-in rather than silently changing
# what every existing frontier_proxy user's requests contain.
_REDACT_PII = os.environ.get("FRONTIER_PROXY_REDACT_PII", "").lower() in ("1", "true", "yes")

app = FastAPI(title="Keystone frontier proxy")
_client = anthropic.AsyncAnthropic()


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": _TARGET_MODEL, "object": "model", "owned_by": "anthropic"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested_model = body.get("model", _TARGET_MODEL)
    system, anthropic_messages = _openai_messages_to_anthropic(body["messages"])

    if _REDACT_PII:
        system, redacted_count = _redact_pii_in_place(system, anthropic_messages)
        if redacted_count:
            logger.info("frontier_proxy.pii_redacted", count=redacted_count)

    # The installed anthropic SDK (1.x)'s AsyncMessages.create() has no
    # temperature/top_p parameter at all (confirmed via
    # inspect.signature — a real, breaking surface change from the 0.x
    # series this project's pyproject.toml pin (">=0.34.0") allows;
    # benchmarks/model_clients.py's own AnthropicClient never passed
    # either for the same reason). Sampling defaults to the API's own.
    kwargs: dict[str, Any] = {
        "model": _TARGET_MODEL,
        "max_tokens": body.get("max_tokens", 4096),
        "messages": anthropic_messages,
    }
    if system:
        kwargs["system"] = system

    response_format = body.get("response_format")
    if response_format and response_format.get("type") == "json_schema":
        schema = response_format["json_schema"]["schema"]
        kwargs["tools"] = [
            {"name": _STRUCTURED_TOOL_NAME, "description": "Emit the structured response.", "input_schema": schema}
        ]
        kwargs["tool_choice"] = {"type": "tool", "name": _STRUCTURED_TOOL_NAME}
    elif body.get("tools"):
        kwargs["tools"] = _openai_tools_to_anthropic(body["tools"])
        choice = _openai_tool_choice_to_anthropic(body.get("tool_choice"))
        if choice:
            kwargs["tool_choice"] = choice

    if not body.get("stream"):
        resp = await _client.messages.create(**kwargs)
        return JSONResponse(_anthropic_response_to_openai(resp, requested_model))

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    if kwargs.get("tools"):
        # Tool-call streaming: the agent loop (src/orchestrator/nodes/coding.py) streams every
        # turn and reassembles tool calls from deltas. Rather than translate Anthropic's
        # input_json_delta events, the complete response is fetched and emitted as one
        # well-formed chunk sequence (content, tool_calls, finish_reason, usage, [DONE]) —
        # correct for every consumer, just not incremental. Incremental translation is W4's.
        resp = await _client.messages.create(**kwargs)
        translated = _anthropic_response_to_openai(resp, requested_model)

        async def emulated_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            base = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": requested_model,
            }

            def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> str:
                choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
                return f"data: {json.dumps({**base, 'choices': [choice]})}\n\n"

            message = translated["choices"][0]["message"]
            if message.get("content"):
                yield chunk({"content": message["content"]})
            if message.get("tool_calls"):
                yield chunk({"tool_calls": [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]})
            yield chunk({}, translated["choices"][0].get("finish_reason") or "stop")
            if include_usage:
                yield f"data: {json.dumps({**base, 'choices': [], 'usage': translated.get('usage', {})})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(emulated_stream(), media_type="text/event-stream")

    async def event_stream():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        async with _client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                delta = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": requested_model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(delta)}\n\n"
        final = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": requested_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        if include_usage:
            # The SDK's stream exposes the final message (with real usage) once the stream is consumed.
            final_message = await stream.get_final_message()
            usage = {
                "prompt_tokens": final_message.usage.input_tokens,
                "completion_tokens": final_message.usage.output_tokens,
                "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
            }
            usage_chunk = {**final, "choices": [], "usage": usage}
            yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    port = int(os.environ.get("FRONTIER_PROXY_PORT", "8090"))
    logger.info("frontier_proxy.starting", target_model=_TARGET_MODEL, port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
