# Use Keystone from VS Code

Two extensions, one gateway, one API key:

- **The Keystone extension** (`vscode/extension/`, below) — the background agent: submit a task, watch its live trace inside VS Code, list the team's tasks, search the team memory.
- **Continue** — chat, inline edit, apply and autocomplete against the same models, plus Keystone's memory tools over MCP. `keystone ide continue-config` generates its config ([section 1](#continue-chat-edit-autocomplete)).

## The Keystone extension

What it does, in one sentence: it turns "assign this to the agent and come back to a PR" into a VS Code command, and shows you everything the agent did while you wait.

- **Submit background task** — a description, the repository (pre-filled from the workspace's git `origin`), and a model picked from `GET /v1/keystone/models` with its live state: READY roles first, an UNHEALTHY / NOT_DEPLOYED / TRAINING role is shown but cannot be picked, `auto` lets the router decide. It is the same `POST /v1/keystone/tasks` the web UI and `curl` use, with the same validation.
- **Open task trace** — the live plan → tool calls → quality gates → review → tests → diff → pull request, streamed from `GET /v1/keystone/tasks/{id}/stream` into a panel. Every tool result and test output in full, expandable. When the task finishes with a PR, a notification offers **Open PR**.
- **List recent tasks** — the tenant's last 50 tasks (status, model role, repository, PR); pick one to open its trace. The status bar shows how many are pending or running.
- **Search team memory** — what the agent would recall for a query (`POST /v1/keystone/memory/recall`, the same ranking the agent's own prompts use).
- **Open the Playground** — the web UI's Playground for the configured gateway.
- **Set API key** — kept in VS Code's secret storage (the OS keychain), never in `settings.json`.

Settings: `keystone.baseUrl` (default `http://localhost:8080`) and `keystone.defaultModel` (default `auto`). The key needs the `agent` scope; add `inference` for the model picker's live states (without it the picker lists the fixed roles as "state unknown").

**Install from the `.vsix`.** CI's `vscode-extension` job attaches `keystone-agents.vsix` to every run; or build it yourself:

```bash
cd vscode/extension && npm ci && npm run build && npm run package
code --install-extension keystone-agents.vsix
```

Then in VS Code: **Keystone: Set API key**, set `keystone.baseUrl` if the gateway is not on `localhost:8080`, and **Keystone: Submit background task**.

**What is verified, and how** (2026-09-22):

- `vscode/extension` unit tests (`npm test`, CI): the SSE decoder, the trace renderer and the model-picker ordering run against **recorded real frames** — `src/__tests__/fixtures/task-stream.sse` is the byte-exact output of the real stream route captured through the FastAPI app (real event publishers, real Redis Stream) and `model-library.json` the real `GET /v1/keystone/models` response with one role served by a real backend process; both come from `scripts/record_task_stream_fixture.py`. Honest scope: the frames carry the shapes each publishing call site emits; they are not a recording of a live model writing code.
- `npm run test:integration` (CI, under xvfb): a real VS Code loads the extension and every command is registered. With `KEYSTONE_URL` and `KEYSTONE_API_KEY` set, it also submits a real task and asserts the trace panel receives real stream events; without them, that one test prints why and skips.
- `scripts/check_ts_types_in_sync.py` (CI lint job, `tests/test_ts_types_in_sync.py`): the `NodeEvent` / `StepEventType` / `StepEvent` / `TaskEvent` declarations in `web/src/api.ts` and `vscode/extension/src/api.ts` are identical, so the two clients cannot drift from `src/orchestrator/events.py` separately.
- The MCP server now also exposes `task_submit` and `task_status` (the same submission function as `POST /v1/keystone/tasks`, tenant-scoped status) — `tests/test_mcp_server.py` runs them over the real MCP transport against a real Postgres, including validation and the auth fail-closed cases.

**What Continue still covers:** chat, inline edit/apply, autocomplete and MCP tool use from the editor — the Keystone extension does not duplicate them.

## Continue: chat, edit, autocomplete

Keystone's gateway is OpenAI-compatible, so the [Continue](https://www.continue.dev/) extension for VS Code talks to it like any other model server — chat, inline edit, apply, and autocomplete — and Continue's MCP client gets Keystone's memory tools (`memory_search`, `memory_add`) and now the task tools (`task_submit`, `task_status`) the same way OpenCode does. Continue's agent mode works too: the gateway passes `tools`, `tool_choice` and `tool` messages through to the model unchanged (`tests/test_messages_api.py` proves the round trip against a real backend), and an extension that speaks the Anthropic Messages API instead can point at `/v1/messages` with the same key.

Verified on: 2026-09-22 — `tests/test_vscode_continue_config.py` parses the generated file and checks every field name against Continue's own config-yaml schema (`packages/config-yaml` in `continuedev/continue`).

## 1. Install Continue

In VS Code: Extensions → search **Continue** → Install. (Or `code --install-extension Continue.continue`.)

## 2. Generate the config

You need the gateway URL and an API key with the `inference` and `agent` scopes (`keystone keys-create`, or the admin API).

```bash
# Prints the YAML; add --output to write it where Continue reads it
keystone ide continue-config --base-url https://keystone.internal:8080 --api-key ks-XXXX-XXXXXXXX

# Write it for VS Code directly
keystone ide continue-config --base-url https://keystone.internal:8080 --api-key ks-XXXX-XXXXXXXX \
  --output ~/.continue/config.yaml

# Air-gapped deployment with the internal CA (pki/): tell Continue to trust it
keystone ide continue-config --base-url https://keystone.internal:8080 --api-key ks-XXXX-XXXXXXXX \
  --ca-bundle /etc/keystone/pki/ca.crt --output ~/.continue/config.yaml
```

`KEYSTONE_API_KEY` in the environment is picked up when `--api-key` is omitted. A checked-in rendering with placeholders is at [`vscode/continue/config.example.yaml`](https://github.com/GGChamp85/keystone/blob/main/vscode/continue/config.example.yaml).

## 3. What the config does

| Continue role | Keystone model | Why |
|---|---|---|
| `chat`, `edit`, `apply`, `summarize` | `coding` (the primary open-weight coder, or whatever you pointed the coding role at — including the frontier proxy) | Quality where it matters |
| `autocomplete` | `coding_fallback` | Cheaper, lower latency, short completions (`maxTokens: 256`, temperature 0) |
| MCP server `Keystone memory` | `GET/POST <gateway>/v1/keystone/mcp` (streamable-HTTP, bearer auth) | The agent's per-repo/per-tenant memory, readable and writable from the IDE |

Model names are the gateway's **roles**, not vendor model ids — swapping the model behind a role (`CODING_MODEL_ID`, a promoted fine-tuned adapter, a frontier proxy) needs no change in VS Code. Per-tenant adapter routing applies to these requests exactly as it does to every other gateway call.

## 4. Try it

- Open any file, select a function, press the Continue edit shortcut and ask for a change — the diff comes from your Keystone deployment.
- In Continue's chat, ask "what conventions does this repo have?" — the agent calls `memory_search` over MCP and answers from Keystone memory.
- Every request lands in the tenant's usage ledger (`GET /v1/keystone/usage`, the web UI's **Spend** view) with the model role it used.

## Background tasks from the IDE

Submitting a full background task (clone → branch → quality gates → tests → PR) and watching its live trace is the Keystone extension above. The OpenCode integration (`docs/deployment/OPENCODE_SETUP.md`) covers the terminal-agent workflow, and any MCP client — Continue included — can call `task_submit` / `task_status` on the gateway's MCP server.
