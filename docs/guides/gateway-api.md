# The gateway API

**What it is.** One HTTPS endpoint, on your network, that your applications and tools call the way they would call a hosted model provider. It speaks the two protocols the ecosystem already uses, the OpenAI chat-completions API and the Anthropic Messages API, and serves open-weight models running on your GPUs. Behind it: API keys, rate limits, token budgets, health-aware routing between models, per-tenant fine-tuned adapters, and a spend ledger.

**Who it is for.** Application teams that want to swap a vendor URL for a self-hosted one; platform teams that want one place to control access, cost and which models serve which workloads.

## Call it from any SDK

Change the base URL; keep the client.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer ks-XXXX-XXXXXXXX" -H "Content-Type: application/json" \
  -d '{"model": "coding", "messages": [{"role": "user", "content": "Write a Python async Redis connection pool"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="ks-XXXX-XXXXXXXX")
client.chat.completions.create(model="coding", messages=[{"role": "user", "content": "..."}], stream=True)
```

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080", api_key="ks-XXXX-XXXXXXXX")
client.messages.create(model="coding", max_tokens=1024, messages=[{"role": "user", "content": "..."}])
```

What passes through unchanged: tool calling (`tools`, `tool_choice`, `tool` messages / `tool_use` and `tool_result` blocks), structured output (`response_format`), streaming (with the backend's real token usage in the final chunk or in `message_delta`). Images and documents are refused with a clear error rather than silently dropped. `POST /v1/messages/count_tokens` returns the gateway's tokenizer count.

## Models are roles

A request names a **role**, not a model file: `coding`, `coding_fallback` or `reasoning`. What serves each role is configuration; the gateway, the agent and every client stay unchanged when it changes. `GET /v1/models` lists the roles; `GET /v1/keystone/models` (the Model Library) adds their live state, the model behind each, its context window and the features it supports.

| Role | Default model | Default GPUs | Used for |
|---|---|---|---|
| `coding` | GLM-5.3-Flash (MIT) | 8 × 80 GB | every agent task and gateway request that asks for it |
| `coding_fallback` | Qwen2.5-Coder-32B-Instruct (Apache-2.0) | 4 × 80 GB | when the primary is unhealthy; simple tasks when complexity routing is on |
| `reasoning` | DeepSeek-R1 (MIT) | 2 × 80 GB | the review and planning steps |

A single 24 GB GPU serving Qwen2.5-Coder-7B is a supported configuration: `keystone init --backend single-gpu` writes it. See [Hardware sizing](../getting-started/hardware-sizing.md).

If a role's endpoint is unhealthy, the router tries the next in its fallback chain; if nothing is healthy, the gateway answers `503` with `Retry-After` and the Model Library shows the breaker state and the last error. Health is probed once and cached (`INFERENCE_HEALTH_CACHE_SECONDS`), so routing adds no per-request probe.

### Change the model behind a role

| Deployment | What changes |
|---|---|
| Docker Compose | `.env`: `VLLM_CODING_MODEL`, `VLLM_CODING_GPU_COUNT`, `VLLM_CODING_MAX_MODEL_LEN`, `VLLM_CODING_TOOL_PARSER` (the compose file reads them) |
| Kubernetes | `helm upgrade ... --set vllm.coding.model=<hf id> --set vllm.coding.servedModelName=<name> --set vllm.coding.toolParser=<parser>` |
| Any hosted OpenAI-compatible endpoint | `VLLM_CODING_URL=<its base url>` and `VLLM_CODING_API_KEY=<its token>`; this is how RunPod Serverless is wired |
| A frontier model, for a role with no GPU | run `python -m benchmarks.frontier_proxy` and point `VLLM_CODING_URL` at it. One vendor is supported today; requests leave your network, and `FRONTIER_PROXY_REDACT_PII=1` redacts personal data first |

Set the matching `CODING_MODEL_ID` so the Model Library and `/v1/models` name the model correctly; it is metadata, never a routing key.

## Scale serving

Each role scales on two independent axes in the Helm chart: `replicas` runs N full instances behind one Service; `nodeCount` runs one instance across several nodes for a model too large for one node's GPUs (vLLM's Ray executor). Optional KEDA autoscaling scales replicas on vLLM's real queue-depth metric. All three render in CI; the multi-node path has not yet been exercised on real multi-node GPU hardware.

## Fine-tuned adapters

A promoted adapter (see [Fine-tune an SLM on your repo](fine-tune-slm-on-your-repo.md)) is served by the base model's role for its tenant automatically: the gateway resolves the tenant's default adapter and sends its name as the model, on the same endpoint, with no client change.

## Index your repositories for retrieval

```bash
keystone ingest https://your-git-host/yourorg/yourrepo --branch main
# 23 file(s) processed, 0 unchanged, 0 skipped, 0 deleted
#   61 chunk(s) created, 61 upserted, 0 stale chunk(s) removed
```

Incremental: an unchanged file is skipped next time; a removed file's chunks are deleted. Chunks are cut on the code's own boundaries (functions and classes, via the real parser for Python) and stored twice: as embeddings in Qdrant and as full-text rows in Postgres. The agent's planning, coding and review steps retrieve with both — vector similarity for the topic, full-text rank for the exact identifiers a task names — fused so a chunk found by both ranks first.

## Keys, limits, spend

- Keys are minted per tenant with scopes (`inference`, `agent`, `finetune`) by an admin (`keystone keys-create`, or `POST /v1/admin/tenants/{id}/keys`) and can be linked to a user with a role (`admin`, `lead`, `developer`). They are stored as a keyed hash (HMAC with the deployment's `VS_SECRET_KEY`), so a copy of the database alone cannot be used to verify a key; **set `VS_SECRET_KEY` once and keep it stable** — production refuses to start without it.
- **Rotate without an outage**: `keystone keys-rotate <tenant> <prefix>` (or `POST /v1/admin/tenants/{id}/keys/{prefix}/rotate`) mints a replacement with the same scopes and limits and keeps the old key valid for 24 hours by default (`grace_hours`, 0 = revoke now). Audited with both prefixes.
- Every limit is `0 = unlimited` by default: request rate, daily and monthly tokens, `max_tokens` ceiling, running agent tasks, and a **monthly dollar budget**. Set any of them after creation with `keystone tenants set-limits <tenant> --monthly-budget-usd 500` (`POST /v1/admin/tenants/{id}/limits`). Once the priced ledger reaches the budget, gateway requests and new agent tasks are refused with `429` and headers `X-VS-Budget-Spent-USD` / `X-VS-Budget-Limit-USD` until the month rolls or the budget is raised.
- Every request and every agent turn lands in the spend ledger: `GET /v1/keystone/usage`, the web UI's **Spend** view, priced by `MODEL_PRICES_PER_MILLION` from your own GPU economics.

## What every response tells you

- `X-Request-ID`: echoed from your request or generated; every log line for that request carries it, so quote it when asking for help.
- `X-VS-Model`: the role that served. `X-VS-Route-Decision`: `requested=…; served=…; model=…; reason=primary|fallback|adapter|cost_routing`, so a fallback or an adapter is never silent.
- `model: "auto"` with `GATEWAY_COST_ROUTING=1` sends a simple last message (a typo, a rename) to the cheaper `coding_fallback` role and everything else to `coding`; the decision is in the header and in the `keystone_route_decisions_total` metric. Off by default.
- Metrics on `/metrics`: time to first token per role, upstream errors per role, routing decisions, budget rejections, tokens per tenant and role.
- `LOG_PROMPTS=1` logs every prompt and reply through the same personal-data redaction the frontier proxy uses. Off by default: prompts are your users' data.

## Verified how

`tests/test_messages_api.py` and `tests/test_completions_adapter_routing.py` run both routes over the real application with the real client talking HTTP to a real backend process, including the installed Anthropic SDK driving the gateway; `tests/e2e/test_ci_backend.py` and `tests/e2e/test_messages_api_ci_backend.py` do the same against the live llama.cpp demo model in CI; `tests/test_inference_health.py` covers fallback and the breaker.
