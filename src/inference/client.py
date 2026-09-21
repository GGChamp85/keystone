# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Inference — Async client for vLLM OpenAI-compatible endpoints.
Handles both streaming and non-streaming completions, token counting,
and automatic retries with exponential backoff.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings

logger = structlog.get_logger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    """
    Connection-level failures are always worth retrying. A response that
    made it back is only worth retrying if it's a 429 (rate limit) or a
    5xx (server error) — a 4xx other than 429 means the request itself was
    bad and retrying it verbatim will just fail the same way again. The
    previous version of this client only retried ConnectError/ReadTimeout,
    silently giving up on transient 429/500s from vLLM's request queue —
    exactly the kind of failure a busy multi-tenant inference server
    produces under load.
    """
    if isinstance(exc, httpx.ConnectError | httpx.ReadTimeout):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return False


class StructuredOutputError(RuntimeError):
    """chat_structured()'s response didn't parse as the requested schema, even after one self-correction retry."""


class InferenceClient:
    """
    Thin async wrapper around a vLLM /v1/chat/completions endpoint.
    One instance per model endpoint.
    """

    def __init__(self, base_url: str, model_id: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    async def close(self) -> None:
        await self._client.aclose()

    # ── Non-streaming completion ──────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,  # surface the real httpx exception+status to callers, not tenacity's RetryError wrapper
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """
        Send a chat completion request and return the full response dict.

        `tools` / `tool_choice`: OpenAI-compatible tool-calling — see
        src/orchestrator/tools/protocol.py's NativeToolProtocol. `response_format`:
        vLLM structured-output (`{"type": "json_schema", "json_schema": {...}}` or
        `{"type": "json_object"}`) — vLLM does not support `tools` and
        `response_format` in the same request (verified against vLLM's own
        OpenAI-compatible server: guided decoding and tool-calling are separate
        code paths), so callers pick one or the other per turn, never both.

        `model_override`: send a different `model` value than this client's
        own `model_id` — e.g. a promoted per-tenant LoRA adapter's served
        name (src/inference/model_router.py's resolve_served_model_name),
        still against this same endpoint (a vLLM instance started with
        `--enable-lora` serves its base model and every one of its
        registered `--lora-modules` names on one endpoint, not separate
        ones per adapter).
        """
        if tools and response_format:
            raise ValueError(
                "tools and response_format are mutually exclusive in one request "
                "(vLLM doesn't support guided decoding + tool-calling together) — "
                "use separate turns instead"
            )

        payload: dict[str, Any] = {
            "model": model_override or self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": False,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if stop:
            payload["stop"] = stop
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format

        t0 = time.monotonic()
        resp = await self._client.post("/chat/completions", json=payload)
        elapsed = time.monotonic() - t0

        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        logger.info(
            "vs_inference.completion",
            model=model_override or self.model_id,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_ms=round(elapsed * 1000),
            has_tool_calls=bool(data.get("choices", [{}])[0].get("message", {}).get("tool_calls")),
        )

        return data

    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        schema_name: str = "response",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        on_usage: Callable[[int, int], None] | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """
        A completion whose content is guaranteed-parseable JSON matching
        `schema` (vLLM structured outputs / guided decoding — backend
        auto-selected by vLLM). On a parse failure, retries exactly once
        with the parser error appended to the conversation so the model can
        self-correct, rather than failing the caller's whole node — this is
        what lets nodes drop the regex-salvage parsing the old prompt-only
        JSON approach needed.

        `on_usage`, if given, is called with `(prompt_tokens, completion_tokens)`
        after every underlying `complete()` call (including a retry attempt) —
        since this method returns only the parsed dict, not the raw response,
        callers that track token spend (e.g. `AgentState.add_tokens`) need this
        hook rather than losing usage accounting entirely.

        `model_override`: see `complete()`'s docstring — passed through
        unchanged on every attempt, including the self-correction retry.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        }
        last_error: Exception | None = None
        current_messages = messages
        for attempt in range(2):
            data = await self.complete(
                current_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                model_override=model_override,
            )
            if on_usage is not None:
                usage = data.get("usage", {})
                on_usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                current_messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": f"That was not valid JSON matching the schema: {exc}. Return ONLY valid JSON.",
                    },
                ]
                logger.warning(
                    "vs_inference.structured_output_retry", model=self.model_id, attempt=attempt, error=str(exc)
                )
        raise StructuredOutputError(f"Model did not return valid JSON matching schema after 2 attempts: {last_error}")

    # ── Streaming completion ──────────────────────────────────

    async def stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Yield SSE chunks from a streaming completion. `model_override`: see
        `complete()`'s docstring — same per-tenant adapter routing applies.
        Each yielded string is a complete `data: {...}` SSE line.
        """
        payload = {
            "model": model_override or self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if stop:
            payload["stop"] = stop

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {chunk}\n\n"

    # ── Health check ──────────────────────────────────────────

    async def health(self) -> bool:
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    # ── List models ───────────────────────────────────────────

    async def list_models(self) -> list[dict]:
        resp = await self._client.get("/models")
        resp.raise_for_status()
        return resp.json().get("data", [])


# ── Singleton registry ────────────────────────────────────────

_clients: dict[str, InferenceClient] = {}


def get_inference_client(role: str) -> InferenceClient:
    """
    Get or create an InferenceClient for the given model role.
    Roles: 'coding', 'coding_fallback', 'reasoning'
    """
    if role not in _clients:
        settings = get_settings()
        endpoints = settings.model_endpoint_map
        model_ids = settings.model_id_map

        if role not in endpoints:
            raise ValueError(f"Unknown model role '{role}'. Available: {list(endpoints.keys())}")

        _clients[role] = InferenceClient(
            base_url=endpoints[role],
            model_id=model_ids[role],
        )

    return _clients[role]


async def close_all_clients() -> None:
    for client in _clients.values():
        await client.close()
    _clients.clear()
