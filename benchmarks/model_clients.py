# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone — Benchmark model clients.

Thin, real clients for getting a single completion from either a
Keystone Inference model (self-hosted, via the same InferenceClient the
production API uses) or a frontier model (Anthropic/OpenAI, via their
official SDKs) — for the eval harness in benchmarks/run_benchmark.py.

Frontier clients are only constructed when the corresponding API key is
present in the environment; the harness treats "not configured" as a skip,
not a failure, since a client running this air-gapped won't have frontier
API access at all — the harness must still work for the self-hosted-only
case.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Protocol


@dataclasses.dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_label: str


class CompletionClient(Protocol):
    async def complete(self, prompt: str) -> CompletionResult: ...


class KeystoneInferenceClient:
    """Wraps the real src.inference.client.InferenceClient — no separate HTTP stack."""

    def __init__(self, role: str = "coding"):
        self.role = role

    async def complete(self, prompt: str) -> CompletionResult:
        from src.inference.client import get_inference_client

        client = get_inference_client(self.role)
        response = await client.complete(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=2048,
        )
        usage = response.get("usage", {})
        return CompletionResult(
            text=response["choices"][0]["message"]["content"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model_label=f"keystone-inference:{self.role}",
        )


class AnthropicClient:
    def __init__(self, model: str = "claude-opus-4-6"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY from env
        return self._client

    async def complete(self, prompt: str) -> CompletionResult:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return CompletionResult(
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            model_label=f"anthropic:{self.model}",
        )


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai

            self._client = openai.AsyncOpenAI()  # reads OPENAI_API_KEY from env
        return self._client

    async def complete(self, prompt: str) -> CompletionResult:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return CompletionResult(
            text=response.choices[0].message.content or "",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            model_label=f"openai:{self.model}",
        )


def available_frontier_clients() -> dict[str, CompletionClient]:
    """
    Only construct clients for providers with an API key actually present —
    this harness must degrade gracefully to self-hosted-only comparison in
    an air-gapped environment with no frontier API access at all.
    """
    clients: dict[str, CompletionClient] = {}
    if os.environ.get("ANTHROPIC_API_KEY"):
        clients["claude-opus"] = AnthropicClient("claude-opus-4-6")
        clients["claude-sonnet"] = AnthropicClient("claude-sonnet-4-6")
    if os.environ.get("OPENAI_API_KEY"):
        clients["gpt-4o"] = OpenAIClient("gpt-4o")
        clients["gpt-4o-mini"] = OpenAIClient("gpt-4o-mini")
    return clients
