# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real tests for benchmarks/frontier_proxy.py — the pure message-translation
functions need no network, but the whole point of this module is that
src.inference.client.InferenceClient (the exact class every real
orchestrator node uses) can talk to it, so the integration tests here hit
a real running instance of the proxy with a real ANTHROPIC_API_KEY and
make real Anthropic API calls through it — no mocking of either side.
"""

from __future__ import annotations

import os

import httpx
import pytest

from benchmarks.frontier_proxy import (
    _anthropic_response_to_openai,
    _openai_messages_to_anthropic,
    _openai_tool_choice_to_anthropic,
    _openai_tools_to_anthropic,
)

pytestmark = pytest.mark.integration

requires_anthropic_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="Needs a real ANTHROPIC_API_KEY"
)

# ── Pure translation functions — no network ─────────────────────────


def test_system_messages_are_concatenated_and_pulled_out():
    system, messages = _openai_messages_to_anthropic(
        [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "system", "content": "Always write tests."},
            {"role": "user", "content": "Add a function."},
        ]
    )
    assert system == "You are a coding agent.\n\nAlways write tests."
    assert messages == [{"role": "user", "content": "Add a function."}]


def test_assistant_tool_calls_translate_to_tool_use_blocks():
    _system, messages = _openai_messages_to_anthropic(
        [
            {"role": "user", "content": "Read app.py"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "function": {"name": "read_file", "arguments": '{"path": "app.py"}'}}],
            },
        ]
    )
    assistant_msg = messages[1]
    assert assistant_msg["role"] == "assistant"
    tool_use = assistant_msg["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["id"] == "call_1"
    assert tool_use["name"] == "read_file"
    assert tool_use["input"] == {"path": "app.py"}


def test_tool_result_message_translates_to_user_tool_result_block():
    _system, messages = _openai_messages_to_anthropic(
        [{"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "def add(a, b): ..."}]
    )
    assert messages[0]["role"] == "user"
    block = messages[0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"
    assert block["content"] == "def add(a, b): ..."


def test_openai_tools_translate_to_anthropic_input_schema():
    tools = _openai_tools_to_anthropic(
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
    )
    assert tools == [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]


def test_forced_tool_choice_translates_correctly():
    assert _openai_tool_choice_to_anthropic({"type": "function", "function": {"name": "read_file"}}) == {
        "type": "tool",
        "name": "read_file",
    }
    assert _openai_tool_choice_to_anthropic("required") == {"type": "any"}
    assert _openai_tool_choice_to_anthropic("auto") is None


class _FakeUsage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeToolUseBlock:
    type = "tool_use"

    def __init__(self, id_, name, input_):
        self.id = id_
        self.name = name
        self.input = input_


class _FakeMessage:
    def __init__(self, content, stop_reason, usage):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage


def test_plain_text_response_translates_to_openai_shape():
    resp = _FakeMessage([_FakeTextBlock("Hello!")], "end_turn", _FakeUsage(10, 5))
    result = _anthropic_response_to_openai(resp, "claude-opus-4-6")
    assert result["choices"][0]["message"]["content"] == "Hello!"
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_real_tool_use_response_translates_to_openai_tool_calls():
    resp = _FakeMessage([_FakeToolUseBlock("toolu_1", "read_file", {"path": "app.py"})], "tool_use", _FakeUsage(20, 8))
    result = _anthropic_response_to_openai(resp, "claude-opus-4-6")
    message = result["choices"][0]["message"]
    assert message["tool_calls"] == [
        {"id": "toolu_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "app.py"}'}}
    ]
    assert result["choices"][0]["finish_reason"] == "tool_calls"


def test_structured_output_tool_use_becomes_json_content_not_a_tool_call():
    """The synthetic emit_structured_response tool call must surface as
    message.content (real JSON text), not as a tool_calls entry — that's
    what makes chat_structured()'s json.loads(content) work unmodified."""
    from benchmarks.frontier_proxy import _STRUCTURED_TOOL_NAME

    resp = _FakeMessage(
        [_FakeToolUseBlock("toolu_1", _STRUCTURED_TOOL_NAME, {"plan": "do the thing"})],
        "tool_use",
        _FakeUsage(20, 8),
    )
    result = _anthropic_response_to_openai(resp, "claude-opus-4-6")
    message = result["choices"][0]["message"]
    assert "tool_calls" not in message
    assert message["content"] == '{"plan": "do the thing"}'


# ── Real end-to-end against a real running proxy + real Anthropic API ──


@pytest.fixture
async def proxy_base_url():
    """Runs the real FastAPI app in-process via ASGITransport — a real
    HTTP round trip through real routing/serialization, just without a
    real socket."""
    from benchmarks.frontier_proxy import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy.test/v1") as client:
        yield client


@requires_anthropic_key
async def test_inference_client_gets_a_real_completion_through_the_proxy(proxy_base_url, monkeypatch):
    """The actual class every orchestrator node uses, talking to the real
    proxy, getting a real Anthropic completion back — proves the whole
    integration, not just the translation functions in isolation."""
    from src.inference.client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client.base_url = "http://proxy.test/v1"
    client.model_id = "claude-opus-4-6"
    client._client = proxy_base_url

    try:
        result = await client.complete(
            messages=[
                {"role": "system", "content": "Reply with exactly one word."},
                {"role": "user", "content": "Say hello."},
            ],
            max_tokens=16,
        )
        content = result["choices"][0]["message"]["content"]
        assert content
        assert result["usage"]["prompt_tokens"] > 0
        assert result["usage"]["completion_tokens"] > 0
    finally:
        pass  # proxy_base_url fixture owns closing the shared httpx client


@requires_anthropic_key
async def test_inference_client_chat_structured_gets_real_schema_conformant_json(proxy_base_url):
    from src.inference.client import InferenceClient

    client = InferenceClient.__new__(InferenceClient)
    client.base_url = "http://proxy.test/v1"
    client.model_id = "claude-opus-4-6"
    client._client = proxy_base_url

    schema = {
        "type": "object",
        "properties": {"greeting": {"type": "string"}},
        "required": ["greeting"],
        "additionalProperties": False,
    }
    result = await client.chat_structured(
        messages=[{"role": "user", "content": "Give me a friendly one-word greeting."}],
        schema=schema,
        max_tokens=64,
    )
    assert isinstance(result, dict)
    assert "greeting" in result
    assert isinstance(result["greeting"], str)


@requires_anthropic_key
async def test_real_tool_call_round_trips_through_the_proxy(proxy_base_url):
    """A real Anthropic tool call, translated to OpenAI tool_calls shape,
    parsed by the real NativeToolProtocol the orchestrator itself uses."""
    from src.orchestrator.tools.protocol import NativeToolProtocol

    resp = await proxy_base_url.post(
        "/chat/completions",
        json={
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "system", "content": "You must call the read_file tool for any file-reading request."},
                {"role": "user", "content": "Read the file named app.py"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file's contents",
                        "parameters": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    },
                }
            ],
            "tool_choice": "required",
            "max_tokens": 256,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]

    protocol = NativeToolProtocol()
    calls = protocol.parse_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"] == "app.py"
