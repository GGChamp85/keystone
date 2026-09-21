# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real integration tests for src/inference/client.py's tool-calling,
structured-output, and retry behavior — against a real HTTP server
(tests/fixtures/fake_vllm_server.py) with scripted responses, not a model
double for `InferenceClient` itself. Verifies the client's own logic
(payload shape, retry-on-429/5xx with the real exception surfaced, the
tools+response_format mutual exclusion guard, chat_structured's
retry-once-on-bad-JSON) — not model behavior, which needs a real LLM.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import httpx
import pytest

from src.inference.client import InferenceClient

# Not marked `integration` — unlike that marker's usual meaning here (needs
# real Postgres/Redis/Qdrant/Docker), this only needs to spawn its own
# throwaway uvicorn subprocess on a free localhost port, so it runs
# unconditionally as a normal test.

_FIXTURE = Path(__file__).parent / "fixtures" / "fake_vllm_server.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def fake_server():
    port = _free_port()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "uvicorn",
        "fake_vllm_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        cwd=str(_FIXTURE.parent),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        async with httpx.AsyncClient() as probe:
            for _ in range(50):
                try:
                    await probe.get(f"{base_url}/last_payload", timeout=0.2)
                    break
                except httpx.ConnectError:
                    await asyncio.sleep(0.1)
            else:
                raise RuntimeError("fake_vllm_server did not start in time")
        yield base_url
    finally:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=5)


async def test_retries_and_recovers_from_429(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={"fail_with_429_times": 2})
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    try:
        resp = await client.complete([{"role": "user", "content": "hi"}])
        assert resp["choices"][0]["message"]["content"] == "hello"
    finally:
        await client.close()


async def test_exhausted_retries_surface_the_real_exception_not_a_retry_wrapper(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={"fail_with_429_times": 10})
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.complete([{"role": "user", "content": "hi"}])
        assert exc_info.value.response.status_code == 429
    finally:
        await client.close()


async def test_tools_and_tool_choice_reach_the_server_and_tool_calls_parse(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    try:
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}]
        resp = await client.complete([{"role": "user", "content": "read app.py"}], tools=tools, tool_choice="auto")
        async with httpx.AsyncClient() as h:
            sent = (await h.get(f"{fake_server}/last_payload")).json()
        assert sent["tools"] == tools
        assert sent["tool_choice"] == "auto"
        assert resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file"
    finally:
        await client.close()


async def test_model_override_replaces_the_sent_model_field(fake_server):
    """model_override (src/inference/model_router.py's per-tenant adapter
    routing) must change what's actually sent as "model", not just be
    accepted and ignored — a promoted adapter's served name has to reach
    vLLM's real /chat/completions request for adapter serving to work."""
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
    client = InferenceClient(base_url=fake_server, model_id="base-model")
    try:
        await client.complete([{"role": "user", "content": "hi"}], model_override="tenant-abc-lora")
        async with httpx.AsyncClient() as h:
            sent = (await h.get(f"{fake_server}/last_payload")).json()
        assert sent["model"] == "tenant-abc-lora"
    finally:
        await client.close()


async def test_no_model_override_falls_back_to_the_clients_own_model_id(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
    client = InferenceClient(base_url=fake_server, model_id="base-model")
    try:
        await client.complete([{"role": "user", "content": "hi"}])
        async with httpx.AsyncClient() as h:
            sent = (await h.get(f"{fake_server}/last_payload")).json()
        assert sent["model"] == "base-model"
    finally:
        await client.close()


async def test_tools_and_response_format_together_is_rejected_client_side(fake_server):
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    try:
        with pytest.raises(ValueError, match="mutually exclusive"):
            await client.complete(
                [{"role": "user", "content": "x"}],
                tools=[{"type": "function", "function": {"name": "x"}}],
                response_format={"type": "json_object"},
            )
    finally:
        await client.close()


async def test_chat_structured_returns_parsed_json(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    try:
        schema = {"type": "object", "properties": {"plan_steps": {"type": "array"}}}
        result = await client.chat_structured([{"role": "user", "content": "plan this"}], schema)
        assert result["plan_steps"] == ["step one", "step two"]
    finally:
        await client.close()


async def test_chat_structured_self_corrects_after_one_bad_json_response(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
        await h.post(f"{fake_server}/set_bad_json_once")
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    try:
        schema = {"type": "object", "properties": {"plan_steps": {"type": "array"}}}
        result = await client.chat_structured([{"role": "user", "content": "plan this"}], schema)
        assert result["plan_steps"] == ["step one", "step two"]
    finally:
        await client.close()


async def test_chat_structured_on_usage_reports_real_token_counts(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    calls: list[tuple[int, int]] = []
    try:
        schema = {"type": "object", "properties": {"plan_steps": {"type": "array"}}}
        await client.chat_structured(
            [{"role": "user", "content": "plan this"}], schema, on_usage=lambda p, c: calls.append((p, c))
        )
        # fake_vllm_server's fixed usage block (see tests/fixtures/fake_vllm_server.py)
        assert calls == [(10, 5)]
    finally:
        await client.close()


async def test_chat_structured_on_usage_called_for_every_retry_attempt(fake_server):
    async with httpx.AsyncClient() as h:
        await h.post(f"{fake_server}/reset", json={})
        await h.post(f"{fake_server}/set_bad_json_once")
    client = InferenceClient(base_url=fake_server, model_id="test-model")
    calls: list[tuple[int, int]] = []
    try:
        schema = {"type": "object", "properties": {"plan_steps": {"type": "array"}}}
        await client.chat_structured(
            [{"role": "user", "content": "plan this"}], schema, on_usage=lambda p, c: calls.append((p, c))
        )
        # One bad-JSON attempt plus the self-correction retry — usage reported for both, not just the last.
        assert calls == [(10, 5), (10, 5)]
    finally:
        await client.close()
