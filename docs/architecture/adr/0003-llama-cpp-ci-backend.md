# ADR 0003 — A real, CPU-only model backend for CI: llama.cpp serving a 0.5B GGUF

**Status**: accepted (2026-09-22)

## Context

Every model-dependent test used to self-skip in CI because no OpenAI-compatible endpoint existed there, so the agent loop, streaming, structured output and tool-call parsing were only ever exercised against a live GPU deployment or a frontier proxy — neither available to a pull request. A fake server (`tests/fixtures/fake_vllm_server.py`) covers the HTTP contract but not a real model's behaviour.

## Decision

CI (and `make up-demo` locally) runs `ghcr.io/ggml-org/llama.cpp:server` with the official Apache-2.0 `Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:Q8_0` (676 MB, cached across runs) as a real OpenAI-compatible backend: `/v1/models`, chat completions, streaming with `stream_options.include_usage`, JSON-schema output, `--jinja` tool templates. `VLLM_CODING_URL` points at it for the whole test job.

## Consequences

- Tests that need a model run for real on every PR (`tests/e2e/test_ci_backend.py`, the agent-loop tests with the text tool protocol, the streaming-usage tests).
- What it does **not** claim: answer quality. A 0.5B model's native `tool_calls` are unreliable (it emits fenced JSON, which `TextToolProtocol` parses); quality numbers come only from the GPU tiers and the persisted benchmark suite.
- llama.cpp and vLLM differ in edge cases (e.g. `/v1/models` fields); anything vLLM-specific is verified on the RunPod tier and stated as such.
