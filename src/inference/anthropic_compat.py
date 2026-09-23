# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — the Anthropic Messages API <-> OpenAI chat-completions translation, both directions.

Two real consumers:

- **Inbound** (`src/api/routes/messages.py`): a client written against the
  Anthropic Messages API (`POST /v1/messages` — the SDKs, agent harnesses
  and IDE tools that speak it) talks to Keystone's self-hosted open-weight
  models. The request is translated to the one wire protocol every backend
  here speaks (OpenAI chat completions, the shape `InferenceClient` sends
  to vLLM / llama.cpp / RunPod), and the reply — including a streamed one —
  is translated back into Anthropic's message and event shapes.
- **Outbound** (`benchmarks/frontier_proxy.py`): the inverse, so the
  unmodified `InferenceClient` can drive a real frontier model through the
  real Anthropic SDK.

Scope, stated plainly: text, tools (tool_use / tool_result, streamed as
`input_json_delta`) and streaming. Images, documents, citations and
batches are refused with a clear `invalid_request_error`, never silently
dropped — a caller must know its image never reached the model.

Everything here is pure (no I/O) and unit-tested; the routes own the I/O.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from src.security.pii_redaction import redact_pii

STRUCTURED_TOOL_NAME = "emit_structured_response"

_OPENAI_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}
_ANTHROPIC_TO_OPENAI_STOP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "stop_sequence": "stop",
}


class UnsupportedContentError(ValueError):
    """A request used a content block or feature this gateway does not translate (images, documents, ...)."""


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


# ── Anthropic request -> OpenAI request (inbound gateway) ─────────────────────


def _text_of(content: Any, *, where: str) -> str:
    """The text of a str-or-blocks content value; anything but text blocks is refused."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text") or "")
        else:
            raise UnsupportedContentError(f"{where}: content block type {btype!r} is not supported by this gateway")
    return "".join(parts)


def anthropic_messages_to_openai(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Anthropic `system` + `messages` -> OpenAI messages. `tool_use` blocks
    become the assistant turn's `tool_calls`; each `tool_result` block
    becomes its own `tool` message (OpenAI's shape: one message per result,
    carrying `tool_call_id`), emitted *before* any text in the same user
    turn because OpenAI expects tool results directly after the call.
    """
    out: list[dict[str, Any]] = []
    system_text = _text_of(system, where="system")
    if system_text:
        out.append({"role": "system", "content": system_text})

    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        where = f"messages[{i}]"
        if role not in ("user", "assistant"):
            raise UnsupportedContentError(f"{where}: role must be 'user' or 'assistant', got {role!r}")

        if isinstance(content, str) or content is None:
            out.append({"role": role, "content": content or ""})
            continue

        if role == "user":
            text_parts: list[str] = []
            for block in content:
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text") or "")
                elif btype == "tool_result":
                    body = _text_of(block.get("content"), where=f"{where}.tool_result")
                    if block.get("is_error"):
                        body = f"[tool error] {body}"
                    out.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": body})
                else:
                    raise UnsupportedContentError(
                        f"{where}: content block type {btype!r} is not supported by this gateway"
                    )
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            continue

        # assistant
        text_parts = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {"name": block.get("name", ""), "arguments": json.dumps(block.get("input") or {})},
                    }
                )
            else:
                raise UnsupportedContentError(f"{where}: content block type {btype!r} is not supported by this gateway")
        assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        out.append(assistant)
    return out


def anthropic_tools_to_openai(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for t in tools:
        if t.get("type") not in (None, "custom"):
            raise UnsupportedContentError(
                f"tool {t.get('name')!r}: server tools of type {t['type']!r} are not supported"
            )
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def anthropic_tool_choice_to_openai(tool_choice: dict[str, Any] | None) -> str | dict[str, Any] | None:
    if not tool_choice:
        return None
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and tool_choice.get("name"):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    raise UnsupportedContentError(f"tool_choice {tool_choice!r} is not supported")


# ── OpenAI response -> Anthropic response (inbound gateway) ───────────────────


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}  # the model emitted malformed JSON; the caller sees exactly what it produced
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def openai_response_to_anthropic(resp: dict[str, Any], model: str) -> dict[str, Any]:
    """A non-streaming OpenAI chat completion -> an Anthropic `Message`."""
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                "name": fn.get("name", ""),
                "input": _parse_arguments(fn.get("arguments")),
            }
        )
    finish = choice.get("finish_reason") or "stop"
    stop_reason = "tool_use" if message.get("tool_calls") else _OPENAI_TO_ANTHROPIC_STOP.get(finish, "end_turn")
    usage = resp.get("usage") or {}
    return {
        "id": new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def sse_event(event_type: str, payload: dict[str, Any]) -> str:
    """One Anthropic-style SSE frame: `event: <type>` + `data: <json>`."""
    return f"event: {event_type}\ndata: {json.dumps({'type': event_type, **payload})}\n\n"


class AnthropicStreamTranslator:
    """
    Feeds OpenAI streaming chunks (already-parsed dicts) and yields
    Anthropic stream events in the exact order the Anthropic SDKs expect:
    message_start, then per content block start / delta / stop (text as
    `text_delta`, tool arguments as `input_json_delta`), then
    message_delta (stop_reason + usage) and message_stop.

    OpenAI backends report token usage only in the final chunk, while
    Anthropic's message_start already carries `input_tokens`: the start
    event carries a tokenizer count of the prompt (`estimate_input_tokens`)
    and the real backend usage, when it arrives, is sent in message_delta's
    `usage` — which is where the SDKs accumulate it, so the final message a
    client assembles has the backend's real numbers.
    """

    def __init__(self, model: str, *, estimate_input_tokens: Callable[[], int]) -> None:
        self.model = model
        self.message_id = new_message_id()
        self._estimate_input_tokens = estimate_input_tokens
        self._started = False
        self._block_index = -1
        self._current_kind: str | None = None  # "text" | "tool_use" | None
        self._openai_call_to_block: dict[int, int] = {}
        self._json_seen: dict[int, bool] = {}  # block index -> any partial_json emitted
        self._finish_reason: str | None = None
        self._saw_tool_calls = False
        self.usage: dict[str, int] | None = None
        self.text_parts: list[str] = []  # for tokenizer-based accounting when the backend sends no usage

    # -- helpers ---------------------------------------------------------------

    def _start(self) -> list[str]:
        self._started = True
        return [
            sse_event(
                "message_start",
                {
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": self._estimate_input_tokens(), "output_tokens": 0},
                    }
                },
            )
        ]

    def _stop_current(self) -> list[str]:
        if self._current_kind is None:
            return []
        events: list[str] = []
        if self._current_kind == "tool_use" and not self._json_seen.get(self._block_index):
            # A tool call with no streamed arguments: the SDKs parse the accumulated partial_json,
            # so an empty input must still arrive as valid JSON.
            events.append(
                sse_event(
                    "content_block_delta",
                    {"index": self._block_index, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
                )
            )
        events.append(sse_event("content_block_stop", {"index": self._block_index}))
        self._current_kind = None
        return events

    def _start_block(self, kind: str, block: dict[str, Any]) -> list[str]:
        events = self._stop_current()
        self._block_index += 1
        self._current_kind = kind
        events.append(sse_event("content_block_start", {"index": self._block_index, "content_block": block}))
        return events

    # -- public ----------------------------------------------------------------

    def feed(self, chunk: dict[str, Any]) -> list[str]:
        events: list[str] = []
        if not self._started:
            events.extend(self._start())
        if chunk.get("usage"):
            u = chunk["usage"]
            self.usage = {
                "input_tokens": int(u.get("prompt_tokens", 0) or 0),
                "output_tokens": int(u.get("completion_tokens", 0) or 0),
            }
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                self.text_parts.append(text)
                if self._current_kind != "text":
                    events.extend(self._start_block("text", {"type": "text", "text": ""}))
                events.append(
                    sse_event(
                        "content_block_delta",
                        {"index": self._block_index, "delta": {"type": "text_delta", "text": text}},
                    )
                )
            for tc in delta.get("tool_calls") or []:
                self._saw_tool_calls = True
                call_index = int(tc.get("index", 0))
                fn = tc.get("function") or {}
                block_index = self._openai_call_to_block.get(call_index)
                if block_index is None or block_index != self._block_index or self._current_kind != "tool_use":
                    if block_index is None:
                        events.extend(
                            self._start_block(
                                "tool_use",
                                {
                                    "type": "tool_use",
                                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                    "name": fn.get("name", ""),
                                    "input": {},
                                },
                            )
                        )
                        self._openai_call_to_block[call_index] = self._block_index
                    else:
                        # deltas for an earlier call after another block started — vLLM never does this,
                        # but a well-formed stream must not lose them
                        events.extend(self._stop_current())
                        self._block_index = block_index
                        self._current_kind = "tool_use"
                if fn.get("arguments"):
                    self._json_seen[self._block_index] = True
                    events.append(
                        sse_event(
                            "content_block_delta",
                            {
                                "index": self._block_index,
                                "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                            },
                        )
                    )
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
        return events

    def finish(self, *, fallback_usage: Callable[[], dict[str, int]] | None = None) -> list[str]:
        """Close the open block and emit message_delta + message_stop. `fallback_usage` supplies the
        tokenizer count when the backend sent no usage chunk."""
        events: list[str] = []
        if not self._started:
            events.extend(self._start())
        events.extend(self._stop_current())
        if self.usage is None and fallback_usage is not None:
            self.usage = fallback_usage()
        finish = self._finish_reason or "stop"
        stop_reason = "tool_use" if self._saw_tool_calls else _OPENAI_TO_ANTHROPIC_STOP.get(finish, "end_turn")
        usage = self.usage or {"input_tokens": 0, "output_tokens": 0}
        events.append(
            sse_event(
                "message_delta",
                {
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]},
                },
            )
        )
        events.append(sse_event("message_stop", {}))
        return events


# ── OpenAI request -> Anthropic request (outbound: the frontier proxy) ────────


def openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Returns (system_prompt, anthropic_messages). OpenAI's `system` role
    has no per-turn position in Anthropic's API — it's one top-level string,
    so every system message is concatenated into it."""
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_parts.append(_text_of(msg.get("content"), where="system"))
            continue

        if role == "user":
            anthropic_messages.append({"role": "user", "content": _text_of(msg.get("content"), where="user")})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.get("content"):
                blocks.append({"type": "text", "text": _text_of(msg["content"], where="assistant")})
            for call in msg.get("tool_calls") or []:
                fn = call.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    tool_input = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                except json.JSONDecodeError:
                    tool_input = {}
                blocks.append({"type": "tool_use", "id": call["id"], "name": fn.get("name", ""), "input": tool_input})
            anthropic_messages.append({"role": "assistant", "content": blocks or ""})
            continue

        if role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_call_id"],
                            "content": _text_of(msg.get("content"), where="tool"),
                        }
                    ],
                }
            )
            continue

        raise ValueError(f"Unsupported message role for the frontier proxy: {role!r}")

    return "\n\n".join(p for p in system_parts if p), anthropic_messages


def redact_pii_in_place(system: str, anthropic_messages: list[dict[str, Any]]) -> tuple[str, int]:
    """
    Redacts real PII patterns (src/security/pii_redaction.py) out of every
    text span actually leaving the network: the system prompt, each user
    turn's content, each assistant text block, and each tool_result's
    content (real file/grep/command output — the highest-risk spot, since
    it's raw content from the caller's own repository). tool_use `input`
    (structured tool-call arguments the model itself generated) is left
    alone — mangling those would break tool-call replay, and they are the
    model's own output, not raw source content. Returns the possibly
    -redacted system string and a total match count for logging.
    """
    total = 0

    def _redact(s: str) -> str:
        nonlocal total
        result = redact_pii(s)
        total += len(result.matches)
        return result.redacted_text

    system = _redact(system) if system else system

    for msg in anthropic_messages:
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _redact(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    block["text"] = _redact(block["text"])
                elif block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                    block["content"] = _redact(block["content"])

    return system, total


def openai_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    converted = []
    for t in tools:
        fn = t["function"]
        converted.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return converted


def openai_tool_choice_to_anthropic(tool_choice: Any) -> dict[str, Any] | None:
    if tool_choice is None or tool_choice == "auto":
        return None  # Anthropic's own default is "auto"
    if tool_choice == "none":
        return None  # caller should have omitted tools instead; nothing to force
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        name = tool_choice.get("function", {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def anthropic_response_to_openai(
    resp: Any, requested_model: str, *, structured_tool_name: str = STRUCTURED_TOOL_NAME
) -> dict[str, Any]:
    """An Anthropic SDK `Message` (duck-typed: `.content` blocks, `.stop_reason`, `.usage`) -> an OpenAI
    chat completion. A call to `structured_tool_name` is the forced-tool emulation of `response_format`
    and comes back as JSON content, not as a tool call."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    structured_content: str | None = None

    for block in resp.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            if block.name == structured_tool_name:
                structured_content = json.dumps(block.input)
            else:
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.input)},
                    }
                )

    finish_reason = _ANTHROPIC_TO_OPENAI_STOP.get(resp.stop_reason or "end_turn", "stop")

    message: dict[str, Any] = {
        "role": "assistant",
        "content": structured_content if structured_content is not None else ("".join(text_parts) or None),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens,
        },
    }
