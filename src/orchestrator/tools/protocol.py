# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — tool-calling protocols.

Two ways a model can emit a tool call, behind one interface so
`nodes/coding.py`'s loop never has to know which one it's talking to:

  - `NativeToolProtocol` — OpenAI-compatible `tool_calls` on the assistant
    message (`tools=`/`tool_choice=` on the request). What GLM-4.6/4.5,
    Qwen2.5-Coder, and most current vLLM tool-parsers support.
  - `TextToolProtocol` — a fenced ```tool_call``` JSON block in plain
    content, for a role with no working native tool-parser (e.g. a custom
    DeepSeek-R1 template). Selected per-role, not globally — see
    `VLLMConfig.tool_protocol` (Phase 0).

Malformed model output (bad JSON in a tool call's arguments, or a
fenced block that isn't valid JSON) is returned as a `ToolCallError`
alongside any well-formed calls in the same turn, never raised — the
loop feeds it back to the model as a tool-result-shaped correction
("your call to X had invalid arguments: ...") so a single hiccup costs
one extra turn, not the whole task.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from src.orchestrator.tools.registry import TOOL_SCHEMAS, render_tools_as_text

_TOOL_CALL_FENCE = re.compile(r"```tool_call\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallError:
    id: str
    name: str
    error: str


ParsedToolCall = ToolCall | ToolCallError


class ToolProtocol(Protocol):
    def system_prompt_suffix(self) -> str:
        """Text to append to the system prompt, or "" if the protocol needs none (native tool_calls need none)."""
        ...

    def request_kwargs(self) -> dict[str, Any]:
        """Extra kwargs for `InferenceClient.complete()` — `tools`/`tool_choice` for native, `{}` for text."""
        ...

    def parse_tool_calls(self, message: dict[str, Any]) -> list[ParsedToolCall]:
        """Every tool call the model made this turn, well-formed or not. Empty list means no tool call — the
        model produced a final answer in `message["content"]`."""
        ...

    def format_tool_result(self, call: ToolCall | ToolCallError, content: str) -> list[dict[str, Any]]:
        """Message(s) to append to the conversation carrying one tool's result back to the model."""
        ...

    def format_assistant_turn(self, message: dict[str, Any], calls: list[ParsedToolCall]) -> dict[str, Any]:
        """The assistant message to append to history for this turn (echoes what the model actually said)."""
        ...


class NativeToolProtocol:
    def __init__(self, schemas: list[dict] = TOOL_SCHEMAS):
        self._schemas = schemas

    def system_prompt_suffix(self) -> str:
        return ""

    def request_kwargs(self) -> dict[str, Any]:
        return {"tools": self._schemas, "tool_choice": "auto"}

    def parse_tool_calls(self, message: dict[str, Any]) -> list[ParsedToolCall]:
        raw_calls = message.get("tool_calls") or []
        parsed: list[ParsedToolCall] = []
        for raw in raw_calls:
            call_id = raw.get("id", "")
            fn = raw.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError as exc:
                parsed.append(ToolCallError(id=call_id, name=name, error=f"arguments were not valid JSON: {exc}"))
                continue
            parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return parsed

    def format_tool_result(self, call: ToolCall | ToolCallError, content: str) -> list[dict[str, Any]]:
        return [{"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}]

    def format_assistant_turn(self, message: dict[str, Any], calls: list[ParsedToolCall]) -> dict[str, Any]:
        turn: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            turn["tool_calls"] = message["tool_calls"]
        return turn


class TextToolProtocol:
    """For a role whose vLLM tool-parser isn't reliable — the model is instructed to reply with a single fenced
    ```tool_call``` JSON block instead of freeform tool_calls, parsed back out of `message["content"]`."""

    def __init__(self, schemas: list[dict] = TOOL_SCHEMAS):
        self._schemas = schemas

    def system_prompt_suffix(self) -> str:
        return "\n\n" + render_tools_as_text(self._schemas)

    def request_kwargs(self) -> dict[str, Any]:
        return {}

    def parse_tool_calls(self, message: dict[str, Any]) -> list[ParsedToolCall]:
        content = message.get("content") or ""
        matches = _TOOL_CALL_FENCE.findall(content)
        if not matches:
            return []
        parsed: list[ParsedToolCall] = []
        for i, block in enumerate(matches):
            call_id = f"text-call-{i}"
            try:
                data = json.loads(block)
            except json.JSONDecodeError as exc:
                parsed.append(ToolCallError(id=call_id, name="", error=f"tool_call block was not valid JSON: {exc}"))
                continue
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if not name:
                parsed.append(ToolCallError(id=call_id, name="", error="tool_call block is missing `name`"))
                continue
            parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))
        return parsed

    def format_tool_result(self, call: ToolCall | ToolCallError, content: str) -> list[dict[str, Any]]:
        name = call.name or "(unknown)"
        return [{"role": "user", "content": f"Result of `{name}`:\n```\n{content}\n```"}]

    def format_assistant_turn(self, message: dict[str, Any], calls: list[ParsedToolCall]) -> dict[str, Any]:
        return {"role": "assistant", "content": message.get("content") or ""}


def get_tool_protocol(protocol: str, schemas: list[dict] = TOOL_SCHEMAS) -> ToolProtocol:
    if protocol == "native":
        return NativeToolProtocol(schemas)
    if protocol == "text":
        return TextToolProtocol(schemas)
    raise ValueError(f"Unknown tool protocol: {protocol!r} (expected 'native' or 'text')")
