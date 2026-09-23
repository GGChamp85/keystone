# Model Library and Playground

One place to see what models this deployment can serve, whether each is up right now, and to try one before pointing an IDE or an app at it. Web UI: **Models** and **Playground** in the top bar; API: `GET /v1/keystone/models`.

## What the library shows

| Group | What it is | State comes from |
|---|---|---|
| **Serving roles** | `coding`, `coding_fallback`, `reasoning` — the ids a client passes as `model` — with the open-weight model behind each, its endpoint and provider | the router's own cached health registry (`src/inference/health.py`): a real probe of the endpoint's `/models`, the breaker's state, the last error. What the library shows is what the next request will hit. |
| **Your adapters** | this tenant's fine-tuned LoRA adapters (`ModelAdapter`), with verdict, rank, path and whether it is the default the router serves this tenant | the base role's state when promoted; `NOT_DEPLOYED` when candidate or retired |
| **Catalog** | the SLM catalog (`src/inference/catalog.py`) — deployable, fine-tunable open-weight coders not behind a role here, with the VRAM to serve and to train | `NOT_DEPLOYED`, or `TRAINING` while a fine-tune job on it runs |

States: **READY** (probed healthy), **UNHEALTHY** (probe failed or breaker open), **NOT_DEPLOYED**, **TRAINING**.

Each entry carries what a decision needs, from where it is actually known: parameters, license and MoE from the catalog; the context window from the endpoint's own `/models` listing when the server reports `max_model_len` (vLLM does), else the catalog, and the entry says which (`context_source`); the API features a client can use through the gateway (`chat_completions`, `anthropic_messages`, `streaming`, `tools`, `json_schema`, `lora_adapters`). `tools` is `null` for a model outside the catalog — unknown, not assumed, because native tool calling needs a vLLM parser for that model's format.

Every card has: **Copy model id** (the exact `model` value to use), **Try in Playground** (READY models only), **Fine-tune** (opens the guided wizard with this base model prefilled), **Deploy** (the exact Helm `--set` lines or `keystone deploy runpod-serverless` command, with the VRAM the model needs).

## Playground

A streamed request to `/v1/chat/completions` — the exact call any OpenAI client makes, nothing special-cased for the UI — against the selected role or promoted adapter, with a system prompt, temperature and max tokens. Every answer shows the backend's **real token usage** (from `stream_options.include_usage`), **time to first token**, total time, tokens per second, the role that served it (`X-VS-Model`, so a fallback is visible) and the finish reason. **Show as curl** prints the request you just made so it can be pasted into a script.

## API

```bash
curl -H "Authorization: Bearer ks-…" http://localhost:8080/v1/keystone/models
```

```json
{
  "generated_at": 1790000000,
  "models": [
    {
      "kind": "role", "id": "coding", "name": "Qwen2.5-Coder-7B-Instruct", "state": "READY",
      "hf_source": "Qwen/Qwen2.5-Coder-7B-Instruct", "provider": "self-hosted vLLM",
      "endpoint": "http://vllm-coding:8000/v1", "served_path": "coding",
      "served_model_ids": ["coding"], "context": 32768, "context_source": "endpoint",
      "params_b": 7.0, "license": "Apache-2.0", "tool_parser": "hermes",
      "deployment": {"vram_serve_gb": 17.5},
      "api_features": {"chat_completions": true, "anthropic_messages": true, "streaming": true, "tools": true, "json_schema": true, "lora_adapters": true},
      "health": {"healthy": true, "breaker": "closed", "consecutive_failures": 0, "last_error": null},
      "adapters": [{"name": "tenant-1a2b3c4d-lora-…", "status": "promoted", "is_default": true, "rank": 8}]
    }
  ]
}
```

Requires the `inference` scope. Probing is cached (`INFERENCE_HEALTH_CACHE_SECONDS`, default 10 s) and shared with the router, so opening the library never adds load beyond what routing already does.

Verified for real: `tests/test_model_library.py` runs the route against a real backend process for one role and dead ports for the others, with a promoted adapter, a retired one and a running fine-tune in a real Postgres.
