# Load test

**What it measures.** Real requests to the gateway's `/v1/chat/completions` under concurrency: latency distribution, failure rate, the backend's real completion tokens per second, and how many requests were served by a fallback role (from the `X-VS-Route-Decision` header). `benchmarks/load/k6_chat.js` is the test; `scripts/k6_smoke.sh` starts the application, mints a tenant and key through the real admin API, runs it, and writes a JSON summary.

## Run it against your deployment

```bash
k6 run -e KEYSTONE_URL=https://keystone.internal -e KEYSTONE_API_KEY=ks-… \
  -e MODEL=coding -e VUS=20 -e DURATION=5m -e MAX_TOKENS=256 -e STREAM=1 \
  -e SUMMARY_JSON=load-$(date +%F).json benchmarks/load/k6_chat.js
```

Publish the summary next to the GPU it ran on (model, GPU count and type, `--max-num-seqs`, context length). The thresholds in the script are the smoke's; tighten them from your own measurements.

## The CI smoke (CPU demo model)

CI runs `scripts/k6_smoke.sh` on every push against the llama.cpp demo model (Qwen2.5-Coder-0.5B on CPU) to keep the script and the gateway's plumbing honest. Its numbers describe a laptop-class CPU, not a GPU deployment, and are recorded here only so the smoke is not a black box. Measured on the development machine on 2026-09-22, 2 virtual users for 20 s, 32 max tokens, non-streaming:

| Requests | Failed | p95 latency | Completion tokens/s (median) | Routed to fallback |
|---|---|---|---|---|
| 34 | 0 | 3.18 s | 36 | 0 |

## What is not published yet

No run against GPU hardware has been recorded. The first RunPod Serverless run (`docs/deployment/RUNPOD_SETUP.md`) will add a row per GPU tier with the exact k6 invocation, so that every number here can be reproduced.
