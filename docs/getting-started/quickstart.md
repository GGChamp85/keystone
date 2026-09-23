# Quickstart — one path

**Outcome:** a running Keystone on your machine, a real model answering through the API and the web UI, and the coding agent fixing a real bug and opening a pull request. **Time:** about 10 minutes for the gateway, 20 more for the agent task.

One sequence of commands from a clone to a real completion and a real coding task. Two tracks differ only in what serves the model: **Track A** needs no GPU (a real 0.5B model on CPU for the gateway, and a frontier model behind the same interface for the coding task — this track is not air-gapped); **Track B** is one 24 GB GPU serving Qwen2.5-Coder-7B, fully self-hosted.

Steps 1 and 3 are executed literally by `tests/e2e/test_golden_path.py` in CI against the same demo model (the `.env` that `keystone init --backend demo-cpu` writes, a tenant and key created through the real admin routes, a real completion through the real router); step 2's compose invocations are the ones `tests/test_cli_ops.py` pins. The steps are true for the commit you have, or CI is red.

## 0. Prerequisites

- Docker with Compose v2 (Docker Desktop on macOS/Windows, Docker Engine on Linux) and ~10 GB free disk.
- Python 3.12+ for the `keystone` CLI: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.
- Track A's coding task: an API key for the frontier model the proxy translates to (`ANTHROPIC_API_KEY`).
- Track B: an NVIDIA GPU with 24 GB and the NVIDIA container toolkit.

## 1. Configure

```bash
git clone https://github.com/GGChamp85/keystone.git && cd keystone
keystone init --backend demo-cpu        # Track A: writes .env with strong random secrets, coding role -> the demo model
# keystone init --backend single-gpu    # Track B: coding role -> vLLM serving Qwen2.5-Coder-7B on your GPU
```

`keystone init` is interactive without `--yes`; it can also configure a git host now (needed only for PR-opening tasks — see step 5).

## 2. Bring it up

```bash
make certs                               # self-signed TLS for a laptop (make certs-ca for anything shared)
keystone up --with-demo-model            # Track A: infra + app + sandbox daemon + the llama.cpp demo model (676 MB, cached)
# keystone up                            # Track B: same, with the vllm-coding service instead (pulls the 7B weights on first start)
keystone doctor                          # every backing service and model endpoint, reported individually
```

`doctor` connects to Postgres, Redis, Qdrant, the sandbox daemon, each model endpoint and the git host directly, independent of the app, and says exactly what is wrong when something is.

## 3. A key, then a real completion

```bash
export KEYSTONE_ROOT_ADMIN_TOKEN=$(grep ^KEYSTONE_ROOT_ADMIN_TOKEN= .env | cut -d= -f2)   # written by keystone init
keystone tenants create my-team you@example.com                                   # prints the tenant id
keystone keys-create <tenant-id> --name laptop --scopes inference,agent,finetune  # prints ks-… once
export KEYSTONE_API_KEY=ks-…

curl -s http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer $KEYSTONE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"coding","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":8}'
```

The response is an OpenAI chat completion with real `usage` from the backend. The same key works for the Anthropic Messages API at `/v1/messages`, and for the web UI at `http://localhost:8080/app/` — open **Models** to see the coding role READY and **Playground** to chat with it.

## 4. Watch the agent write code

Track A points the coding role at a frontier model for this step (the 0.5B demo model is for plumbing, not for solving tasks). In a second shell:

```bash
FRONTIER_PROXY_MODEL=claude-opus-4-6 python -m benchmarks.frontier_proxy     # any real model string of that vendor
keystone init --backend frontier-proxy --yes                                  # rewrites .env: coding role -> the proxy (secrets regenerate too — see note)
docker compose up -d app worker                                               # pick up the new .env
```

`keystone init` writes a fresh `.env` (new secrets included); to only re-point the role, edit `VLLM_CODING_URL=http://host.docker.internal:8090/v1` in the existing `.env` instead. The seeded repo task needs a git host the agent may push to — a local Gitea works (`docker run -d --name gitea -p 3000:3000 gitea/gitea:1.22`, create a bot user and token, then `GIT_ALLOWED_HOSTS`/`GIT_HOST_API_URL`/`GIT_HOST_TOKEN` in `.env`; `keystone init`'s git-host prompt writes exactly those). Then submit it and stream its trace:

```bash
python -m benchmarks.agent_runner --task token_bucket --backend-label frontier    # clones, plans, edits with tools, runs tests, pushes, opens a PR
```

The trace in the web UI shows every tool call, diff, test run and quality finding as it happens. Track B runs the same command against the 7B model on your GPU.

## 5. Unhappy paths you should see

- A repository host that is not in `GIT_ALLOWED_HOSTS` is refused before any clone, with the host named in the error.
- No healthy model endpoint: the gateway answers 503 with `Retry-After`, and **Models** shows the role UNHEALTHY with the breaker state and the last error.
- A wrong or missing API key: 401, never a silent fallback.

## 6. Next

- [Hardware sizing](hardware-sizing.md) for the GPU behind each role.
- [Fine-tune an SLM on your repo](../guides/fine-tune-slm-on-your-repo.md): describe → plan & cost → approve → verdict → served in seconds.
- [Use it from VS Code](../guides/use-from-vscode.md).
- Production: [Kubernetes](../deployment/KUBERNETES_CLIENT_VPC.md), [RunPod Serverless](../deployment/RUNPOD_SETUP.md), [air-gapped](../airgap/OFFLINE_INSTALL_RUNBOOK.md).
