# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real end-to-end checks of the CI model backend — the `demo-model` compose
profile / CI job step: llama.cpp's server serving the official
Qwen2.5-Coder-0.5B-Instruct GGUF on CPU, reached through the exact
`InferenceClient` every gateway route and orchestrator node uses. No mocks:
these make real HTTP calls to a real model server and assert on what it
really returned.

What this proves, and only this: that the OpenAI-compatible plumbing —
health check, non-streaming completion with real token usage, streaming —
works end to end against a real backend, so every model-dependent test in
this repo can run for real in CI instead of self-skipping. A 0.5B model's
answer quality and tool-call reliability are deliberately NOT asserted
here; that is what the GPU tiers and the benchmark suite measure.

Self-skips when VLLM_CODING_URL is unset or unreachable — the same
"real infra required" convention as tests/test_git_workflow_integration.py.
"""

from __future__ import annotations

import os

import httpx
import pytest

from src.inference.client import InferenceClient

pytestmark = pytest.mark.integration

_BACKEND_URL = os.environ.get("VLLM_CODING_URL", "")
_MODEL_ID = os.environ.get("CODING_MODEL_ID", "demo-coder")


def _backend_reachable() -> bool:
    if not _BACKEND_URL:
        return False
    try:
        return httpx.get(f"{_BACKEND_URL.rstrip('/')}/models", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        return False


requires_model_backend = pytest.mark.skipif(
    not _backend_reachable(),
    reason="Needs a reachable OpenAI-compatible model backend at VLLM_CODING_URL (see docker-compose.yml demo-model)",
)


@pytest.fixture
async def client():
    c = InferenceClient(base_url=_BACKEND_URL, model_id=_MODEL_ID)
    try:
        yield c
    finally:
        await c.close()


@requires_model_backend
async def test_health_check_passes_against_the_real_backend(client):
    assert await client.health() is True


@requires_model_backend
async def test_real_completion_returns_content_and_real_usage(client):
    result = await client.complete(
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=16,
        temperature=0.0,
    )
    content = result["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content.strip(), result
    usage = result["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@requires_model_backend
async def test_real_stream_yields_chunks_and_terminates(client):
    chunks = [chunk async for chunk in client.stream(messages=[{"role": "user", "content": "Say hi."}], max_tokens=16)]
    assert chunks, "stream produced no chunks"
    assert chunks[-1].strip().endswith("[DONE]"), chunks[-1]
